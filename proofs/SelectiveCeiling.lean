import Mathlib

/-!
Finite versions of the selective ceiling (Proposition prop:ceiling of the article).
`p x` is the probability of input `x`, `η x y` the conditional probability of label `y` given `x`,
`m η x` its largest value. Probabilities enter only through nonnegativity.
-/

open Finset

variable {X Y : Type*} [Fintype X] [Fintype Y] [Nonempty Y]

/-- Largest conditional probability of a label given `x`. -/
noncomputable def m (η : X → Y → ℝ) (x : X) : ℝ :=
  (univ : Finset Y).sup' univ_nonempty (η x)

lemma le_m (η : X → Y → ℝ) (x : X) (y : Y) : η x y ≤ m η x :=
  le_sup' (η x) (mem_univ y)

/-- (i) Ceiling: for any decision `d` and accepted set `A`, the mass of correct accepted decisions is at most
the mass of `m` over `A`. Dividing both sides by `P(A)` gives the conditional statement. -/
theorem ceiling (p : X → ℝ) (hp : ∀ x, 0 ≤ p x) (η : X → Y → ℝ) (d : X → Y) (A : Finset X) :
    ∑ x ∈ A, p x * η x (d x) ≤ ∑ x ∈ A, p x * m η x :=
  sum_le_sum fun x _ => mul_le_mul_of_nonneg_left (le_m η x (d x)) (hp x)

/-- (ii) Information: if `η x y = ∑_w q x w * ζ x w y` with nonnegative weights, observing `w` as well can only
raise the largest conditional probability. -/
theorem information {W : Type*} [Fintype W] (q : X → W → ℝ) (hq : ∀ x w, 0 ≤ q x w)
    (ζ : X → W → Y → ℝ) (x : X) :
    m (fun x y => ∑ w, q x w * ζ x w y) x ≤ ∑ w, q x w * m (fun (xw : X × W) y => ζ xw.1 xw.2 y) (x, w) := by
  unfold m
  refine sup'_le _ _ fun y _ => ?_
  exact sum_le_sum fun w _ => mul_le_mul_of_nonneg_left (le_sup' (fun y => ζ x w y) (mem_univ y)) (hq x w)

/-- (iii) Coarsening: merging labels by `g` can only raise the largest conditional probability. -/
theorem coarsening {Y' : Type*} [Fintype Y'] [Nonempty Y'] [DecidableEq Y'] (η : X → Y → ℝ)
    (hη : ∀ x y, 0 ≤ η x y) (g : Y → Y') (x : X) :
    m η x ≤ m (fun x y' => ∑ y ∈ univ.filter (fun y => g y = y'), η x y) x := by
  unfold m
  refine sup'_le _ _ fun y _ => ?_
  refine le_trans ?_ (le_sup' (fun y' => ∑ z ∈ univ.filter (fun z => g z = y'), η x z) (mem_univ (g y)))
  exact single_le_sum (fun z _ => hη x z) (mem_filter.mpr ⟨mem_univ y, rfl⟩)
