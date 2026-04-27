//! Sentence embedding pooling helpers.

use ndarray::{Array2, Array3};

const EPSILON: f32 = 1.0e-12;

pub(crate) fn mean_pool_with_mask(
    last_hidden_state: &Array3<f32>,
    attention_mask: &Array2<i64>,
) -> Array2<f32> {
    let (batch, seq, hidden) = last_hidden_state.dim();
    let mut pooled = Array2::<f32>::zeros((batch, hidden));

    for b in 0..batch {
        let mut count = 0.0_f32;
        for j in 0..seq {
            let mask = attention_mask[(b, j)];
            if mask != 0 {
                count += 1.0;
                for d in 0..hidden {
                    pooled[(b, d)] += last_hidden_state[(b, j, d)];
                }
            }
        }

        let divisor = count.max(1.0);
        for d in 0..hidden {
            pooled[(b, d)] /= divisor;
        }
    }

    pooled
}

pub(crate) fn l2_normalize_rows(rows: &mut Array2<f32>) {
    for mut row in rows.rows_mut() {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > EPSILON {
            for value in &mut row {
                *value /= norm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, arr3};

    fn assert_array_close(actual: &Array2<f32>, expected: &Array2<f32>) {
        assert_eq!(actual.dim(), expected.dim());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1.0e-6);
        }
    }

    #[test]
    fn mean_pool_all_ones_mask_is_sequence_mean() {
        let hidden = arr3(&[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]);
        let mask = arr2(&[[1, 1, 1]]);

        let pooled = mean_pool_with_mask(&hidden, &mask);

        assert_array_close(&pooled, &arr2(&[[3.0, 4.0]]));
    }

    #[test]
    fn mean_pool_mixed_mask_excludes_masked_positions() {
        let hidden = arr3(&[
            [
                [1.0, 2.0, 3.0, 4.0],
                [10.0, 20.0, 30.0, 40.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            [
                [2.0, 4.0, 6.0, 8.0],
                [4.0, 8.0, 12.0, 16.0],
                [100.0, 100.0, 100.0, 100.0],
            ],
        ]);
        let mask = arr2(&[[1, 0, 1], [1, 1, 0]]);

        let pooled = mean_pool_with_mask(&hidden, &mask);

        assert_array_close(
            &pooled,
            &arr2(&[[3.0, 4.0, 5.0, 6.0], [3.0, 6.0, 9.0, 12.0]]),
        );
    }

    #[test]
    fn mean_pool_all_zero_mask_returns_zero_vector() {
        let hidden = arr3(&[[[1.0, 2.0], [3.0, 4.0]]]);
        let mask = arr2(&[[0, 0]]);

        let pooled = mean_pool_with_mask(&hidden, &mask);

        assert_array_close(&pooled, &arr2(&[[0.0, 0.0]]));
    }

    #[test]
    fn l2_normalize_unit_vector_is_itself() {
        let mut rows = arr2(&[[0.6, 0.8]]);

        l2_normalize_rows(&mut rows);

        assert_array_close(&rows, &arr2(&[[0.6, 0.8]]));
    }

    #[test]
    fn l2_normalize_scaled_vector_becomes_unit_vector() {
        let mut rows = arr2(&[[3.0, 4.0]]);

        l2_normalize_rows(&mut rows);

        assert_array_close(&rows, &arr2(&[[0.6, 0.8]]));
    }

    #[test]
    fn l2_normalize_zero_vector_stays_zero() {
        let mut rows = arr2(&[[0.0, 0.0, 0.0]]);

        l2_normalize_rows(&mut rows);

        assert_array_close(&rows, &arr2(&[[0.0, 0.0, 0.0]]));
    }
}
