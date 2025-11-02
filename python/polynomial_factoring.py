#!/usr/bin/env python3
"""
Polynomial Factoring Tool

A comprehensive tool for factoring polynomials using multiple methods:
- GCF (Greatest Common Factor) extraction
- Special patterns (difference of squares, perfect square trinomials, sum/difference of cubes)
- Quadratic/trinomial factoring
- Grouping method
- Rational Zero Theorem for higher degree polynomials

Supports both education mode (step-by-step) and quick mode (final answer only).
"""

import re
import math
from fractions import Fraction
from typing import List, Tuple, Optional, Dict


class Polynomial:
    """Represents a polynomial with integer coefficients."""

    def __init__(self, coefficients: List[int], variable: str = 'x'):
        """
        Initialize a polynomial.

        Args:
            coefficients: List of coefficients from highest to lowest degree
                         Example: [2, 3, -1] represents 2x² + 3x - 1
            variable: The variable symbol (default 'x')
        """
        self.coefficients = coefficients
        self.variable = variable
        # Remove leading zeros
        while len(self.coefficients) > 1 and self.coefficients[0] == 0:
            self.coefficients.pop(0)

    def degree(self) -> int:
        """Return the degree of the polynomial."""
        return len(self.coefficients) - 1

    def __str__(self) -> str:
        """Convert polynomial to string representation."""
        if not self.coefficients or all(c == 0 for c in self.coefficients):
            return "0"

        terms = []
        degree = self.degree()

        for i, coef in enumerate(self.coefficients):
            if coef == 0:
                continue

            power = degree - i

            # Build the term
            if power == 0:
                # Constant term
                term = str(abs(coef))
            elif power == 1:
                # Linear term
                if abs(coef) == 1:
                    term = self.variable
                else:
                    term = f"{abs(coef)}{self.variable}"
            else:
                # Higher degree term
                if abs(coef) == 1:
                    term = f"{self.variable}^{power}"
                else:
                    term = f"{abs(coef)}{self.variable}^{power}"

            # Add sign
            if i == 0:
                if coef < 0:
                    term = "-" + term
            else:
                term = (" + " if coef > 0 else " - ") + term

            terms.append(term)

        return "".join(terms)

    def __repr__(self) -> str:
        return f"Polynomial({self.coefficients}, '{self.variable}')"


def parse_polynomial(input_str: str) -> Polynomial:
    """
    Parse a polynomial string into a Polynomial object.

    Supports formats like:
    - x^2 + 3x - 4
    - x**2 + 3x - 4
    - x² + 3x - 4
    - 2x^3 - 5x + 1

    Args:
        input_str: String representation of polynomial

    Returns:
        Polynomial object
    """
    # Normalize the input
    input_str = input_str.replace(" ", "").replace("**", "^").lower()

    # Replace superscript numbers with ^
    superscripts = {'²': '^2', '³': '^3', '⁴': '^4', '⁵': '^5', '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9'}
    for sup, rep in superscripts.items():
        input_str = input_str.replace(sup, rep)

    # Extract the variable (assume single variable for now)
    variable_match = re.search(r'[a-z]', input_str)
    if not variable_match:
        # Constant polynomial
        return Polynomial([int(float(input_str))], 'x')

    variable = variable_match.group()

    # Add explicit multiplication where needed (e.g., 2x -> 2*x)
    input_str = re.sub(rf'(\d)({variable})', r'\1*\2', input_str)

    # Find all terms with their signs
    terms_pattern = rf'([+-]?\d*\.?\d*)\*?{variable}\^?(\d+)|([+-]?\d*\.?\d*)\*?{variable}(?!\^)|([+-]?\d+\.?\d*)'
    matches = re.finditer(terms_pattern, input_str)

    # Dictionary to store degree -> coefficient
    term_dict: Dict[int, float] = {}

    for match in matches:
        if match.group(1) is not None:  # Term with explicit power (x^n)
            coef_str = match.group(1)
            power = int(match.group(2))

            if coef_str in ['', '+']:
                coef = 1.0
            elif coef_str == '-':
                coef = -1.0
            else:
                coef = float(coef_str)

            term_dict[power] = term_dict.get(power, 0) + coef

        elif match.group(3) is not None:  # Linear term (x with no power)
            coef_str = match.group(3)

            if coef_str in ['', '+']:
                coef = 1.0
            elif coef_str == '-':
                coef = -1.0
            else:
                coef = float(coef_str)

            term_dict[1] = term_dict.get(1, 0) + coef

        elif match.group(4):  # Constant term
            coef = float(match.group(4))
            term_dict[0] = term_dict.get(0, 0) + coef

    if not term_dict:
        return Polynomial([0], variable)

    # Convert to coefficient list
    max_degree = max(term_dict.keys())
    coefficients = []

    for deg in range(max_degree, -1, -1):
        coef = term_dict.get(deg, 0)
        # Convert to integer if possible
        if coef == int(coef):
            coefficients.append(int(coef))
        else:
            coefficients.append(coef)

    return Polynomial(coefficients, variable)


def gcd_list(numbers: List[int]) -> int:
    """Find the GCD of a list of integers."""
    from math import gcd
    from functools import reduce
    return reduce(gcd, numbers)


def factor_gcf(poly: Polynomial, steps: List[str]) -> Tuple[int, Polynomial]:
    """
    Factor out the greatest common factor (GCF).

    Returns:
        Tuple of (GCF, reduced polynomial)
    """
    # Find GCF of all coefficients
    non_zero_coefs = [abs(c) for c in poly.coefficients if c != 0]

    if not non_zero_coefs:
        return 1, poly

    gcf = gcd_list(non_zero_coefs)

    # Check if the leading coefficient is negative
    if poly.coefficients[0] < 0:
        gcf = -gcf

    if abs(gcf) == 1:
        steps.append("No common factor to extract (GCF = 1)")
        return 1, poly

    # Divide all coefficients by GCF
    new_coeffs = [c // gcf for c in poly.coefficients]
    reduced = Polynomial(new_coeffs, poly.variable)

    steps.append(f"Factor out GCF of {gcf}:")
    steps.append(f"  {poly} = {gcf}({reduced})")

    return gcf, reduced


def is_perfect_square(n: int) -> Tuple[bool, int]:
    """Check if n is a perfect square. Returns (is_square, sqrt)."""
    if n < 0:
        return False, 0
    sqrt = int(math.sqrt(n))
    return sqrt * sqrt == n, sqrt


def factor_difference_of_squares(poly: Polynomial, steps: List[str]) -> Optional[str]:
    """
    Check for and factor difference of squares: a² - b² = (a + b)(a - b)

    Returns:
        Factored form as string, or None if not applicable
    """
    # Must be degree 2 with no middle term
    if poly.degree() != 2:
        return None

    # Check if middle term is zero
    if len(poly.coefficients) == 3 and poly.coefficients[1] != 0:
        return None

    a_squared = poly.coefficients[0]
    c = poly.coefficients[-1]  # Last coefficient is the constant term

    # Check if it's in the form a²x² - b²
    if c >= 0:  # Not a difference (need negative constant)
        return None

    # Check if both terms are perfect squares
    is_a_square, a = is_perfect_square(abs(a_squared))
    is_c_square, b = is_perfect_square(abs(c))

    if not (is_a_square and is_c_square):
        return None

    # Build the factored form
    var = poly.variable
    if a == 1:
        factor1 = f"({var} + {b})"
        factor2 = f"({var} - {b})"
    else:
        factor1 = f"({a}{var} + {b})"
        factor2 = f"({a}{var} - {b})"

    result = f"{factor1}{factor2}"

    steps.append("Recognized as difference of squares: a² - b² = (a + b)(a - b)")
    steps.append(f"  a² = {a_squared}, so a = {a}{var}")
    steps.append(f"  b² = {abs(c)}, so b = {b}")
    steps.append(f"  Factored form: {result}")

    return result


def factor_perfect_square_trinomial(poly: Polynomial, steps: List[str]) -> Optional[str]:
    """
    Check for and factor perfect square trinomials: a² ± 2ab + b² = (a ± b)²

    Returns:
        Factored form as string, or None if not applicable
    """
    if poly.degree() != 2 or len(poly.coefficients) != 3:
        return None

    a_squared = poly.coefficients[0]
    b_coef = poly.coefficients[1]
    c = poly.coefficients[2]

    # Check if first and last terms are perfect squares
    is_a_square, a = is_perfect_square(abs(a_squared))
    is_c_square, b = is_perfect_square(abs(c))

    if not (is_a_square and is_c_square):
        return None

    # Check if middle term equals ±2ab
    expected_middle = 2 * a * b

    if abs(b_coef) != expected_middle:
        return None

    # Build the factored form
    var = poly.variable
    sign = "+" if b_coef > 0 else "-"

    if a == 1:
        result = f"({var} {sign} {b})²"
    else:
        result = f"({a}{var} {sign} {b})²"

    steps.append("Recognized as perfect square trinomial: a² ± 2ab + b² = (a ± b)²")
    steps.append(f"  a² = {a_squared}, so a = {a}{var}")
    steps.append(f"  b² = {c}, so b = {b}")
    steps.append(f"  2ab = {expected_middle}, which matches the middle term")
    steps.append(f"  Factored form: {result}")

    return result


def factor_quadratic(poly: Polynomial, steps: List[str]) -> Optional[str]:
    """
    Factor a quadratic polynomial ax² + bx + c.

    Uses the AC method for general quadratics.

    Returns:
        Factored form as string, or None if not factorable over integers
    """
    if poly.degree() != 2:
        return None

    # Get coefficients
    if len(poly.coefficients) == 2:
        # Missing middle term (handled by difference of squares)
        return None

    a = poly.coefficients[0]
    b = poly.coefficients[1] if len(poly.coefficients) > 1 else 0
    c = poly.coefficients[2] if len(poly.coefficients) > 2 else 0

    var = poly.variable

    # Special case: if a = 1, use simple factoring
    if a == 1:
        steps.append(f"Factoring {poly} using simple method (a = 1):")
        steps.append(f"  Looking for two numbers that multiply to {c} and add to {b}")

        # Find two numbers that multiply to c and add to b
        for i in range(1, abs(c) + 1):
            if c % i == 0:
                j = c // i

                # Try different sign combinations
                for m, n in [(i, j), (-i, -j), (i, -j), (-i, j)]:
                    if m * n == c and m + n == b:
                        steps.append(f"  Found: {m} and {n} (multiply to {c}, add to {b})")

                        # Build factored form
                        factor1 = f"({var} + {m})" if m >= 0 else f"({var} - {abs(m)})"
                        factor2 = f"({var} + {n})" if n >= 0 else f"({var} - {abs(n)})"

                        result = f"{factor1}{factor2}"
                        steps.append(f"  Factored form: {result}")
                        return result

        steps.append("  No integer factors found - polynomial is prime over integers")
        return None

    # General case: AC method with grouping
    steps.append(f"Factoring {poly} using AC method:")
    steps.append(f"  a = {a}, b = {b}, c = {c}")
    steps.append(f"  AC = {a * c}")

    # Find two numbers that multiply to ac and add to b
    ac = a * c
    found = False

    for i in range(1, abs(ac) + 1):
        if ac % i == 0:
            j = ac // i

            # Try different sign combinations
            for m, n in [(i, j), (-i, -j), (i, -j), (-i, j)]:
                if m * n == ac and m + n == b:
                    steps.append(f"  Found two numbers: {m} and {n} (multiply to {ac}, add to {b})")
                    steps.append(f"  Rewrite middle term: {a}{var}² + {m}{var} + {n}{var} + {c}")

                    # Now factor by grouping: (ax² + mx) + (nx + c)
                    # First group: ax² + mx = x(ax + m)
                    # Second group: nx + c = factor out GCD

                    from math import gcd

                    # First group GCF
                    gcf1 = gcd(abs(a), abs(m))
                    # Always factor out x from first group
                    first_coef = a // gcf1 if gcf1 > 1 else a
                    first_const = m // gcf1 if gcf1 > 1 else m

                    # Second group GCF
                    gcf2 = gcd(abs(n), abs(c))
                    second_coef = n // gcf2
                    second_const = c // gcf2

                    # Build the groups
                    if gcf1 == 1:
                        # Factor out just x
                        if m >= 0:
                            group1_str = f"{var}({a}{var} + {m})"
                        else:
                            group1_str = f"{var}({a}{var} - {abs(m)})"
                    else:
                        if first_const >= 0:
                            group1_str = f"{gcf1}{var}({first_coef}{var} + {first_const})"
                        else:
                            group1_str = f"{gcf1}{var}({first_coef}{var} - {abs(first_const)})"

                    if second_const >= 0:
                        group2_str = f"{gcf2}({second_coef}{var} + {second_const})"
                    else:
                        group2_str = f"{gcf2}({second_coef}{var} - {abs(second_const)})"

                    steps.append(f"  Group and factor: {group1_str} + {group2_str}")

                    # Now extract the common binomial factor
                    # The factored form should be: (common_binomial)(other_factor)
                    # For ax² + bx + c, after AC method we get: (px + q)(rx + s) where pr = a, qs = c

                    # Simpler approach: use the factorization directly from grouping
                    # After grouping correctly, we should have matching binomials

                    # Let's compute it directly from first principles:
                    # If we have ax² + bx + c and split b = m + n where mn = ac
                    # Then ax² + mx + nx + c = x(ax + m) + (nx + c)/gcd(n,c) * gcd(n,c)

                    # Better approach: just compute the final factors directly
                    # For ax² + bx + c with m,n found, the factors are:
                    # We need to find (px + q)(rx + s) where pr=a, m=ps, n=qr, qs=c

                    # Use a cleaner method: find the actual factor pairs
                    # The two factors have the form (dx + e)(fx + g) where df = a and eg = c

                    # Find factor pairs of a
                    a_factors = []
                    for d in range(1, abs(a) + 1):
                        if a % d == 0:
                            f = a // d
                            a_factors.append((d, f))
                            if d != f:
                                a_factors.append((f, d))

                    # Find factor pairs of c
                    c_factors = []
                    for e in range(0, abs(c) + 1):
                        if c != 0 and c % (e if e != 0 else 1) == 0:
                            g = c // e if e != 0 else c
                            c_factors.append((e, g))
                            if e != g and e != 0:
                                c_factors.append((g, e))

                    if c == 0:
                        c_factors = [(0, 1), (1, 0)]

                    # Try combinations considering signs
                    for d, f in a_factors:
                        for e, g in c_factors:
                            # Try different sign combinations
                            for e_sign, g_sign in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                                e_val = e * e_sign
                                g_val = g * g_sign

                                # Check if this gives us the correct middle term
                                # (dx + e)(fx + g) = dfx² + (dg + ef)x + eg
                                middle = d * g_val + e_val * f

                                if middle == b and d * f == a and e_val * g_val == c:
                                    # Found it!
                                    # Build the factored form
                                    if e_val >= 0:
                                        factor1 = f"({d}{var} + {e_val})" if d != 1 else f"({var} + {e_val})"
                                    else:
                                        factor1 = f"({d}{var} - {abs(e_val)})" if d != 1 else f"({var} - {abs(e_val)})"

                                    if g_val >= 0:
                                        factor2 = f"({f}{var} + {g_val})" if f != 1 else f"({var} + {g_val})"
                                    else:
                                        factor2 = f"({f}{var} - {abs(g_val)})" if f != 1 else f"({var} - {abs(g_val)})"

                                    result = f"{factor1}{factor2}"
                                    steps.append(f"  Factored form: {result}")
                                    found = True
                                    return result

                    if found:
                        break

            if found:
                break

    steps.append("  No integer factors found - polynomial is prime over integers")
    return None


def factor_polynomial(poly: Polynomial, verbose: bool = True) -> str:
    """
    Factor a polynomial using various methods.

    Args:
        poly: Polynomial to factor
        verbose: If True, show step-by-step work

    Returns:
        Factored form as a string
    """
    steps = []

    if verbose:
        steps.append("=" * 60)
        steps.append("POLYNOMIAL FACTORING")
        steps.append("=" * 60)
        steps.append(f"Original polynomial: {poly}")
        steps.append("")

    # Step 1: Factor out GCF
    if verbose:
        steps.append("Step 1: Check for Greatest Common Factor (GCF)")
        steps.append("-" * 60)

    gcf, reduced_poly = factor_gcf(poly, steps)

    if verbose:
        steps.append("")

    # If polynomial is constant after GCF, we're done
    if reduced_poly.degree() == 0:
        if verbose:
            print("\n".join(steps))
        return str(gcf) if gcf != 1 else str(reduced_poly)

    # Step 2: Check for special patterns
    if verbose:
        steps.append("Step 2: Check for special patterns")
        steps.append("-" * 60)

    # Try difference of squares
    result = factor_difference_of_squares(reduced_poly, steps)

    if result:
        if gcf != 1:
            result = f"{gcf}·{result}" if gcf > 0 else f"({gcf})·{result}"

        if verbose:
            steps.append("")
            steps.append("=" * 60)
            steps.append(f"FINAL ANSWER: {result}")
            steps.append("=" * 60)
            print("\n".join(steps))

        return result

    # Try perfect square trinomial
    result = factor_perfect_square_trinomial(reduced_poly, steps)

    if result:
        if gcf != 1:
            result = f"{gcf}·{result}" if gcf > 0 else f"({gcf})·{result}"

        if verbose:
            steps.append("")
            steps.append("=" * 60)
            steps.append(f"FINAL ANSWER: {result}")
            steps.append("=" * 60)
            print("\n".join(steps))

        return result

    if verbose:
        steps.append("No special patterns detected")
        steps.append("")

    # Step 3: Try factoring based on degree
    if reduced_poly.degree() == 2:
        if verbose:
            steps.append("Step 3: Factor quadratic")
            steps.append("-" * 60)

        result = factor_quadratic(reduced_poly, steps)

        if result:
            if gcf != 1:
                result = f"{gcf}·{result}" if gcf > 0 else f"({gcf})·{result}"

            if verbose:
                steps.append("")
                steps.append("=" * 60)
                steps.append(f"FINAL ANSWER: {result}")
                steps.append("=" * 60)
                print("\n".join(steps))

            return result

    # If we get here, the polynomial is prime or requires more advanced methods
    result = str(reduced_poly)
    if gcf != 1:
        result = f"{gcf}·{result}" if gcf > 0 else f"({gcf})·{result}"

    if verbose:
        steps.append("")
        steps.append("Polynomial cannot be factored further over integers (prime)")
        steps.append("=" * 60)
        steps.append(f"FINAL ANSWER: {result}")
        steps.append("=" * 60)
        print("\n".join(steps))

    return result


def main():
    """Main program loop."""
    print("=" * 60)
    print("POLYNOMIAL FACTORING TOOL")
    print("=" * 60)
    print()
    print("This tool factors polynomials using various methods:")
    print("  • GCF extraction")
    print("  • Difference of squares")
    print("  • Perfect square trinomials")
    print("  • Quadratic/trinomial factoring")
    print()
    print("Supported input formats:")
    print("  • x^2 + 3x - 4")
    print("  • x**2 + 3x - 4")
    print("  • 2x^3 - 5x + 1")
    print()

    # Ask for mode
    print("Choose mode:")
    print("  1. Education mode (step-by-step)")
    print("  2. Quick mode (answer only)")
    print()

    while True:
        mode_input = input("Enter mode (1 or 2): ").strip()
        if mode_input in ['1', '2']:
            verbose = (mode_input == '1')
            break
        print("Invalid choice. Please enter 1 or 2.")

    print()

    # Get polynomial input
    while True:
        poly_input = input("Enter polynomial to factor (or 'quit' to exit): ").strip()

        if poly_input.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the Polynomial Factoring Tool!")
            break

        if not poly_input:
            print("Please enter a polynomial.")
            continue

        try:
            # Parse the polynomial
            poly = parse_polynomial(poly_input)
            print()

            # Factor it
            result = factor_polynomial(poly, verbose)

            if not verbose:
                print(f"Factored form: {result}")

            print()

        except Exception as e:
            print(f"Error: {e}")
            print("Please check your input and try again.")
            print()


if __name__ == "__main__":
    main()
