#define SQRT_MAGIC_F 0x5f3759df
inline float fastsqrt(const float x){
    const float xhalf = 0.5f * x;

    union {
        float x;
        int i;
    } u;
    u.x = x;
    u.i = SQRT_MAGIC_F - (u.i >> 1);
    return x * u.x * (1.5f - xhalf * u.x * u.x);
}   // fast implementation of sqrt

inline double fastpow(double a, const double b) {
    int exp = (int)b;

    union {
        double d;
        int x[2];
    } u = { a };
    u.x[1] = (int)((b - exp) * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;

    double r = 1.0;
    while (exp) {
        if (exp & 1) {
            r *= a;
        }
        a *= a;
        exp >>= 1;
    }

    return r * u.d;
} // fast implementation of pow
