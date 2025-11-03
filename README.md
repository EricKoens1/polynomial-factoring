# Polynomial Factoring Tool

A comprehensive Python tool for factoring polynomials using multiple methods, with educational step-by-step explanations and automatic verification.

## Features

✅ **Multiple Factoring Methods**
- Greatest Common Factor (GCF) extraction
- Difference of squares: a² - b² = (a + b)(a - b)
- Perfect square trinomials: a² ± 2ab + b² = (a ± b)²
- Sum/difference of cubes: a³ ± b³
- Quadratic factoring (simple and AC method)
- Grouping method for 4-term polynomials
- Rational Zero Theorem with synthetic division for higher degree polynomials

✅ **Dual Factorization Display**
- Shows factorization over **integers/reals** (standard form)
- Shows complete factorization over **complex numbers** (with complex roots)
- Helps understand when polynomials are "fully factored"

✅ **Automatic Verification**
- Expands factored form back to original polynomial
- Verifies correctness with ✓ or ✗ indicator
- Provides confidence in results

✅ **Educational Features**
- **Education mode**: Shows step-by-step work with explanations
- **Quick mode**: Just shows final answer
- Clear, formatted output for learning

✅ **Flexible Input**
- Supports multiple polynomial formats: `x^2`, `x**2`, `x²`
- Handles coefficients, spaces, and various notations

## Installation

### Prerequisites
- Python 3.6 or higher

### Setup
```bash
# Clone the repository
git clone https://github.com/EricKoens1/polynomial-factoring.git
cd polynomial-factoring

# No additional dependencies required - uses Python standard library only!
```

## Usage

### Running the Program

```bash
python3 python/polynomial_factoring.py
```

### Interactive Mode

The program will prompt you to:
1. **Choose a mode:**
   - `1` for Education mode (step-by-step)
   - `2` for Quick mode (answer only)

2. **Enter polynomials** to factor

3. **Type 'quit'** to exit

### Example Session

```
============================================================
POLYNOMIAL FACTORING TOOL
============================================================

Choose mode:
  1. Education mode (step-by-step)
  2. Quick mode (answer only)

Enter mode (1 or 2): 1

Enter polynomial to factor (or 'quit' to exit): x^3 + 8

============================================================
POLYNOMIAL FACTORING
============================================================
Original polynomial: x^3 + 8

Step 1: Check for Greatest Common Factor (GCF)
------------------------------------------------------------
No common factor to extract (GCF = 1)

Step 2: Check for special patterns
------------------------------------------------------------
Recognized as sum of cubes: a³ + b³ = (a + b)(a² - ab + b²)
  a³ = 1, so a = 1x
  b³ = 8, so b = 2
  Factored form: (x + 2)(x² - 2x + 4)

============================================================
FACTORED FORM (over integers): (x + 2)(x² - 2x + 4)

VERIFICATION: ✓ Factorization is correct!

FACTORED FORM (over complex numbers): (x + 2)(x - (1 + √3i))(x - (1 - √3i))
============================================================
```

## Supported Input Formats

The tool accepts polynomials in various formats:

| Format | Example |
|--------|---------|
| Caret notation | `x^2 + 3x - 4` |
| Double asterisk | `x**2 + 3x - 4` |
| Unicode superscript | `x² + 3x - 4` |
| Implicit multiplication | `2x^3 - 5x + 1` |
| With/without spaces | `x^2+5x+6` or `x^2 + 5x + 6` |

## Examples

### Simple Quadratics

**Input:** `x^2 + 5x + 6`

**Output:**
```
FACTORED FORM (over integers): (x + 2)(x + 3)
VERIFICATION: ✓ Factorization is correct!
```

---

### Complex Quadratics (AC Method)

**Input:** `3x^2 + 7x + 4`

**Output:**
```
FACTORED FORM (over integers): (x + 1)(3x + 4)
VERIFICATION: ✓ Factorization is correct!
```

---

### GCF + Difference of Squares

**Input:** `4x^2 - 16`

**Output:**
```
Factor out GCF of 4:
  4x^2 - 16 = 4(x^2 - 4)

Recognized as difference of squares: a² - b² = (a + b)(a - b)
  a² = 1, so a = 1x
  b² = 4, so b = 2

FACTORED FORM (over integers): 4·(x + 2)(x - 2)
VERIFICATION: ✓ Factorization is correct!
```

---

### Sum of Cubes

**Input:** `x^3 + 8`

**Output:**
```
FACTORED FORM (over integers): (x + 2)(x² - 2x + 4)
VERIFICATION: ✓ Factorization is correct!

FACTORED FORM (over complex numbers): (x + 2)(x - (1 + √3i))(x - (1 - √3i))
```

---

### Difference of Cubes

**Input:** `8x^3 - 27`

**Output:**
```
FACTORED FORM (over integers): (2x - 3)(4x² + 6x + 9)
VERIFICATION: ✓ Factorization is correct!
```

---

### Grouping Method

**Input:** `x^3 + 3x^2 + 2x + 6`

**Output:**
```
Attempting factoring by grouping (4 terms):
  Group 1: x³ + 3x² = x²(x + 3)
  Group 2: 2x + 6 = 2(x + 3)
  Common binomial factor found: (x + 3)

FACTORED FORM (over integers): (x + 3)(x² + 2)
VERIFICATION: ✓ Factorization is correct!
```

---

### Rational Zero Theorem

**Input:** `x^3 - 6x^2 + 11x - 6`

**Output:**
```
Using Rational Zero Theorem to find possible rational zeros
  Possible rational zeros: 1, 2, 3, 6, -1, -2, -3, -6
  Actual zeros found: 1, 2, 3
  Divided by (x - 1), quotient: x² - 5x + 6
  Divided by (x - 2), quotient: x - 3
  Divided by (x - 3), quotient: 1

FACTORED FORM (over integers): (x - 1)(x - 2)(x - 3)
VERIFICATION: ✓ Factorization is correct!
```

---

### Irreducible Quadratics (Complex Roots)

**Input:** `x^2 + 4`

**Output:**
```
FACTORED FORM (over integers): x² + 4
(Polynomial is prime over integers)

FACTORED FORM (over complex numbers): (x - 2i)(x + 2i)
```

## Algorithm Overview

### Factoring Process

The tool follows a systematic approach:

1. **GCF Extraction** - Factor out any common factors first
2. **Special Pattern Recognition** - Check for:
   - Difference of squares
   - Perfect square trinomials
   - Sum/difference of cubes
3. **Degree-Based Factoring:**
   - **Degree 2**: Quadratic factoring (simple or AC method)
   - **Degree 3+**: Try grouping, then Rational Zero Theorem
4. **Verification** - Expand and verify the factorization
5. **Complex Factorization** - Show complete factorization if applicable

### How Verification Works

1. Parse the factored form string
2. Extract GCF multipliers and individual factors
3. Multiply all factors together using polynomial multiplication
4. Compare result to original polynomial
5. Display ✓ if correct, ✗ if incorrect

### Complex Factorization

When a quadratic has a negative discriminant (b² - 4ac < 0):
- It has no real roots
- It's irreducible over the integers/reals
- But it CAN be factored over complex numbers using the quadratic formula

The tool automatically detects these cases and shows both forms.

## What You'll Learn

### From Each Implementation

**General Concepts:**
- How to recognize different factoring patterns
- When polynomials are "fully factored" (depends on number system!)
- The relationship between factors and roots
- How complex numbers extend factorization

**Specific Methods:**
- **GCF**: Finding common factors
- **Difference of Squares**: Recognizing a² - b² patterns
- **Perfect Squares**: Identifying (a ± b)² patterns
- **Sum/Difference of Cubes**: Using cube root formulas
- **AC Method**: Factoring complex quadratics
- **Grouping**: Strategic pairing for 4-term polynomials
- **Rational Zero Theorem**: Systematic approach for higher degrees

## Project Structure

```
polynomial-factoring/
├── README.md              # This file
├── python/
│   └── polynomial_factoring.py  # Main Python implementation
└── .gitignore
```

## Future Enhancements

Possible additions for the future:

- [ ] Add more language implementations (C, JavaScript, etc.)
- [ ] Handle polynomials with fractional coefficients
- [ ] Factor polynomials with multiple variables
- [ ] Add graphing capabilities to visualize roots
- [ ] Export results to LaTeX format
- [ ] Handle higher degree polynomials (degree 4+) more systematically
- [ ] Add partial fraction decomposition
- [ ] Interactive web interface

## Technical Details

### Time Complexity

| Method | Complexity |
|--------|------------|
| GCF Extraction | O(n) where n = number of terms |
| Pattern Recognition | O(1) - constant time checks |
| Quadratic Factoring | O(c) where c = constant term |
| Rational Zero Theorem | O(p×q) where p, q are factors of coefficients |

### Space Complexity

O(n) where n is the degree of the polynomial (for storing coefficients)

## Educational Use

This tool is perfect for:

- **Algebra students** learning factoring techniques
- **Teachers** demonstrating different factoring methods
- **Self-learners** practicing polynomial factoring
- **Homework verification** - check your answers!
- **Understanding complex numbers** - see how factorization extends

## Contributing

Contributions are welcome! Some ideas:

1. Add implementations in other programming languages
2. Improve complex number formatting
3. Add more factoring methods
4. Enhance error messages
5. Add unit tests

## License

This project is open source and available for educational purposes.

## Author

Created as a learning project to explore:
- Polynomial algebra and factoring algorithms
- Multi-method problem solving
- Educational software design
- Complex number mathematics

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

**Happy Factoring!** 🎓✨
