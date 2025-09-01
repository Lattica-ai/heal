#pragma once

#include <vector>
#include <cstdint>
#include <iostream>

class Num {
public:
    typedef uint64_t word;
    typedef void (*random_func)(uint8_t *bytes, size_t n_bytes);

    Num() = default;
    Num(size_t n, word w, bool neg = false);
    Num(const word *a, const word *b, bool neg = false);
    Num(const Num &a);
    Num& operator = (const Num &a);
    Num(int i);
    Num(int64_t i);
    Num(const char *c, word base = 10, char **endptr = NULL);

    static Num random_bits(size_t n_bits, random_func func);
    static Num random_inclusive(const Num &inclusive, random_func func);
    static Num random_exclusive(const Num &exclusive, random_func func);
    static Num random_second_exclusive(const Num &inclusive_min_val, const Num &exclusive_max_val, random_func func);
    static Num random_both_inclusive(const Num &inclusive_min_val, const Num &inclusive_max_val, random_func func);

    void resize(size_t n);
    void pop_back();
    void push_back(word b);
    word& back();
    const word& back() const;
    size_t size() const;
    word& operator [] (size_t i);
    const word& operator [] (size_t i) const;
    Num& set_neg(bool _neg);
    Num& truncate();
    size_t bitlength() const;
    size_t count_trailing_zeros() const;

    void print(
        std::vector<char> &text,
        word base = 10,
        const char *alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    ) const;

    friend std::ostream& operator << (std::ostream& out, const Num &num);
    double to_double() const;
    bool can_convert_to_int(int *result);
    Num pow(size_t exponent) const;
    Num mod_pow(Num exponent, const Num &modulus) const;
    Num sqrt() const;

    Num& operator >>= (size_t n_bits);
    Num& operator <<= (size_t n_bits);
    Num& set_bit(size_t i);
    word get_bit(size_t i) const;
    void clr_bit(size_t i);
    Num& mul_word(word b);
    Num& add_word(word carry, size_t i0 = 0);

    Num& operator ++();
    Num& operator += (const Num &b);
    Num& operator -= (const Num &b);
    Num& operator *= (const Num &b);
    Num& operator /= (const Num &b);
    Num& operator %= (const Num &b);

    bool operator == (const Num &b) const;
    bool operator != (const Num &b) const;
    bool operator <= (const Num &b) const;
    bool operator >= (const Num &b) const;
    bool operator <  (const Num &b) const;
    bool operator >  (const Num &b) const;

    Num operator + (const Num &b) const;
    Num operator - (const Num &b) const;
    Num operator * (const Num &b) const;
    Num operator / (const Num &b) const;
    Num operator % (const Num &b) const;
    Num operator - () const;

    Num operator >> (size_t n_bits) const;
    Num operator << (size_t n_bits) const;

private:
    std::vector<word> words;
    bool neg = false;

    static word word_mask();
    static size_t word_bits();
    static word word_half_mask();
    static word char_to_word(char c);
    static word word_gcd(word a, word b);
    static int cmp_abs(const Num &a, const Num &b);
    static int cmp(const Num &a, const Num &b);
    static size_t word_bitlength(word a);
    static size_t word_count_trailing_zeros(word a);
    static word add_carry(word *a, word b);
    static word sub_carry(word *a, word b);
    static word word_mul_hi(word a, word b);
    static Num& add_unsigned_overwrite(Num &a, const Num &b);
    static Num& sub_unsigned_overwrite(Num &a, const Num &b);
    static Num mul_long(const Num &a, const Num &b);
    static Num mul_karatsuba(const Num &a, const Num &b);
    static Num mul(const Num &a, const Num &b);
    static Num add_signed(const Num &a, bool a_neg, const Num &b, bool b_neg);
    static void div_mod(const Num &numerator, Num denominator, Num &quotient, Num &remainder);
    static void div_mod_half_word(const Num &numerator, word denominator, Num &quotient, word &remainder);
    static void split(const Num &a, Num *parts, size_t n_parts, size_t n);
    static Num div(const Num &numerator, const Num& denominator);
    static Num mod(const Num &numerator, const Num& denominator);
    static Num add_unsigned(const Num &a, const Num &b);
    static Num sub_unsigned(const Num &a, const Num &b);
    static Num add(const Num &a, const Num &b);
    static Num sub(const Num &a, const Num &b);
    static Num gcd(const Num &a0, const Num &b0);
};
