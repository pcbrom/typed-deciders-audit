import Mathlib

/-!
Finite versions of the sufficiency conditions of the dynamic gate (Proposition prop:dyngate of the article).
`w b` is the share of items falling in bin `b` (nonnegative), `θ b` the accuracy of bin `b`, `A` the accepted bins.
-/

open Finset

variable {B : Type*} [Fintype B]

/-- (i) Aggregation: if every accepted bin has accuracy at least `α`, the accuracy among accepted items, the
`w`-weighted mean of `θ` over `A`, is at least `α` (stated without division, so it also covers an empty `A`). -/
theorem aggregation (w θ : B → ℝ) (hw : ∀ b, 0 ≤ w b) (α : ℝ) (A : Finset B) (h : ∀ b ∈ A, α ≤ θ b) :
    α * ∑ b ∈ A, w b ≤ ∑ b ∈ A, w b * θ b := by
  rw [mul_sum]
  exact sum_le_sum fun b hb => by
    rw [mul_comm]
    exact mul_le_mul_of_nonneg_left (h b hb) (hw b)

/-- (ii) Joint credibility, finite form of the union bound: if each accepted bin falls below `α` with probability at
most `q / |A|`, the probability that some accepted bin does is at most `q`. Here `p b` stands for those probabilities. -/
theorem joint_credibility (A : Finset B) (p : B → ℝ) (q : ℝ) (hA : 0 < A.card)
    (h : ∀ b ∈ A, p b ≤ q / A.card) : ∑ b ∈ A, p b ≤ q := by
  calc ∑ b ∈ A, p b ≤ ∑ _b ∈ A, q / A.card := sum_le_sum h
    _ = q := by
      rw [sum_const, nsmul_eq_mul]
      field_simp

/-- (ii) as stated in the article: every bin gated at level `q / n`, with `n` the number of bins, so that
`|A| ≤ n`; then the probability that some accepted bin falls below `α` is at most `q`. -/
theorem joint_credibility_bins (A : Finset B) (p : B → ℝ) (q : ℝ) (n : ℕ) (hq : 0 ≤ q) (hn : 0 < n)
    (hA : A.card ≤ n) (h : ∀ b ∈ A, p b ≤ q / n) : ∑ b ∈ A, p b ≤ q := by
  calc ∑ b ∈ A, p b ≤ ∑ _b ∈ A, q / n := sum_le_sum h
    _ = A.card * (q / n) := by rw [sum_const, nsmul_eq_mul]
    _ ≤ n * (q / n) := by
      apply mul_le_mul_of_nonneg_right _ (div_nonneg hq (Nat.cast_nonneg n))
      exact_mod_cast hA
    _ = q := by
      have : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
      field_simp
