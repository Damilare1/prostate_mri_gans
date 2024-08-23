import functools
import sys
import tensorflow as tf
import tensorflow_probability as tfp
class Eval:
    def _symmetric_matrix_square_root(self, mat, eps=1e-10):
        """Compute square root of a symmetric matrix.

        Note that this is different from an elementwise square root. We want to
        compute M' where M' = sqrt(mat) such that M' * M' = mat.

        Also note that this method **only** works for symmetric matrices.

        Args:
            mat: Matrix to take the square root of.
            eps: Small epsilon such that any element less than eps will not be square
            rooted to guard against numerical instability.

        Returns:
            Matrix square root of mat.
        """
        # Unlike numpy, tensorflow's return order is (s, u, v)
        s, u, v = tf.linalg.svd(mat)
        # sqrt is unstable around 0, just use 0 in such case
        si = tf.compat.v1.where(tf.less(s, eps), s, tf.sqrt(s))
        # Note that the v returned by Tensorflow is v = V
        # (when referencing the equation A = U S V^T)
        # This is unlike Numpy which returns v = V^T
        return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)


    def _trace_sqrt_product(self, sigma, sigma_v):
        """Find the trace of the positive sqrt of product of covariance matrices.

        '_symmetric_matrix_square_root' only works for symmetric matrices, so we
        cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
        ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

        Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
        We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
        Note the following properties:
        (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
            => eigenvalues(A A B B) = eigenvalues (A B B A)
        (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
            => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
        (iii) forall M: trace(M) = sum(eigenvalues(M))
            => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                        = sum(sqrt(eigenvalues(A B B A)))
                                        = sum(eigenvalues(sqrt(A B B A)))
                                        = trace(sqrt(A B B A))
                                        = trace(sqrt(A sigma_v A))
        A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
        use the _symmetric_matrix_square_root function to find the roots of these
        matrices.

        Args:
            sigma: a square, symmetric, real, positive semi-definite covariance matrix
            sigma_v: same as sigma

        Returns:
            The trace of the positive square root of sigma*sigma_v
        """

        # Note sqrt_sigma is called "A" in the proof above
        sqrt_sigma = self._symmetric_matrix_square_root(sigma)

        # This is sqrt(A sigma_v A) above
        sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))

        return tf.linalg.trace(self._symmetric_matrix_square_root(sqrt_a_sigmav_a))


    def _frechet_classifier_distance_from_activations_helper(
        self, activations1, activations2, streaming=False):
        """A helper function evaluating the frechet classifier distance."""
        activations1 = tf.convert_to_tensor(value=activations1)
        activations1.shape.assert_has_rank(2)
        activations2 = tf.convert_to_tensor(value=activations2)
        activations2.shape.assert_has_rank(2)

        activations_dtype = activations1.dtype
        if activations_dtype != tf.float64:
            activations1 = tf.cast(activations1, tf.float64)
            activations2 = tf.cast(activations2, tf.float64)

        # Compute mean and covariance matrices of activations.
        m = (tf.reduce_mean(input_tensor=activations1, axis=0),)
        m_w = (tf.reduce_mean(input_tensor=activations2, axis=0),)
        # Calculate the unbiased covariance matrix of first activations.
        num_examples_real = tf.cast(tf.shape(input=activations1)[0], tf.float64)
        sigma = (num_examples_real / (num_examples_real - 1) *
                    tfp.stats.covariance(activations1),)
        # Calculate the unbiased covariance matrix of second activations.
        num_examples_generated = tf.cast(tf.shape(input=activations2)[0], tf.float64)
        sigma_w = (num_examples_generated / (num_examples_generated - 1) *
                    tfp.stats.covariance(activations2),)
            
        # m, m_w, sigma, sigma_w are tuples containing one or two elements: the first
        # element will be used to calculate the score value and the second will be
        # used to create the update_op. We apply the same operation on the two
        # elements to make sure their value is consistent.

        def _calculate_fid(m, m_w, sigma, sigma_w):
            """Returns the Frechet distance given the sample mean and covariance."""
            # Find the Tr(sqrt(sigma sigma_w)) component of FID
            sqrt_trace_component = self._trace_sqrt_product(sigma, sigma_w)

            # Compute the two components of FID.

            # First the covariance component.
            # Here, note that trace(A + B) = trace(A) + trace(B)
            trace = tf.linalg.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

            # Next the distance between means.
            mean = tf.reduce_sum(input_tensor=tf.math.squared_difference(
                m, m_w))  # Equivalent to L2 but more stable.
            fid = trace + mean
            if activations_dtype != tf.float64:
              fid = tf.cast(fid, activations_dtype)
            return fid

        result = tuple(
            _calculate_fid(m_val, m_w_val, sigma_val, sigma_w_val)
            for m_val, m_w_val, sigma_val, sigma_w_val in zip(m, m_w, sigma, sigma_w))
        if streaming:
            return result
        else:
            return result[0]


    def _frechet_classifier_distance_helper(self, input_tensor1,
                                            input_tensor2,
                                            classifier_fn,
                                            num_batches=1,
                                            streaming=False):
      """A helper function for evaluating the frechet classifier distance."""
      input_list1 = tf.split(input_tensor1, num_or_size_splits=num_batches)
      input_list2 = tf.split(input_tensor2, num_or_size_splits=num_batches)

      stack1 = tf.stack(input_list1)
      stack2 = tf.stack(input_list2)

      # Compute the activations using the memory-efficient `map_fn`.
      def compute_activations(elems):
          return tf.map_fn(
              fn=classifier_fn,
              elems=elems,
              parallel_iterations=1,
              back_prop=False,
              swap_memory=True,
              name='RunClassifier')

      activations1 = compute_activations(stack1)
      activations2 = compute_activations(stack2)

      # Ensure the activations have the right shapes.
      activations1 = tf.concat(tf.unstack(activations1), 0)
      activations2 = tf.concat(tf.unstack(activations2), 0)

      return self._frechet_classifier_distance_from_activations_helper(
              activations1, activations2, streaming=streaming)

    def _frechet_classifier_distance(self, input_tensor1,
                                    input_tensor2,
                                    classifier_fn,
                                    num_batches=1):
        return self._frechet_classifier_distance_helper(
            input_tensor1,
            input_tensor2,
            classifier_fn,
            num_batches,
            streaming=False)
    
    def _classifier_fn(self):
      def classifier_fn(images):
        # view_images(images)
        output = images[0]
        x = tf.nest.map_structure(lambda x: tf.keras.layers.Flatten(data_format='channels_last')(x), output)
        x = tf.cast(x, dtype=tf.float64)  
        return x
      return classifier_fn

    def get_fid(self, real_image, gen_image):
        frechet_inception_distance = functools.partial(
            self._frechet_classifier_distance,
            classifier_fn=self._classifier_fn())
        num_batches = 30 
    
        fid = frechet_inception_distance(real_image, gen_image, num_batches=num_batches)
        return fid
    
sys.modules[__name__] = Eval