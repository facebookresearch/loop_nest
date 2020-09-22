#pragma once

#include <limits>

inline void baseline_CW_HWC(unsigned IOC, unsigned OHW, unsigned KHW,
                            float const* AData, float const* BData,
                            float* CData, int alpha = 0)
{
    unsigned IHW = OHW + KHW - 1;

    for (unsigned oh = 0; oh < OHW; ++oh)
    {
        for (unsigned ow = 0; ow < OHW; ++ow)
        {
            for (unsigned ioc = 0; ioc < IOC; ++ioc)
            {
                float& c = CData[ioc + ow * IOC + oh * OHW * IOC];

                if (alpha == 0)
                {
                    c = 0.f;
                }

                for (unsigned kh = 0; kh < KHW; ++kh)
                {
                    unsigned ih = oh + kh;

                    for (unsigned kw = 0; kw < KHW; ++kw)
                    {
                        unsigned iw = ow + kw;

                        c += AData[ioc + iw * IOC + ih * IOC * IHW] *
                             BData[ioc + kw * IOC + kh * IOC * KHW];
                    }
                }
            }
        }
    }
}

inline void baseline_padded_CW_HWC(unsigned IOC, unsigned OHW, unsigned KHW,
                                   unsigned HW_PAD, float const* AData,
                                   float const* BData, float* CData,
                                   int alpha = 0)
{
    unsigned IHW = OHW + KHW - 1 - 2 * HW_PAD;

    for (unsigned oh = 0; oh < OHW; ++oh)
    {
        for (unsigned ow = 0; ow < OHW; ++ow)
        {
            for (unsigned ioc = 0; ioc < IOC; ++ioc)
            {
                float& c = CData[ioc + ow * IOC + oh * OHW * IOC];

                if (alpha == 0)
                {
                    c = 0.f;
                }

                for (unsigned kh = 0; kh < KHW; ++kh)
                {
                    unsigned ih = oh + kh;

                    if (ih >= HW_PAD && (ih - HW_PAD < IHW))
                    {
                        ih -= HW_PAD;
                        for (unsigned kw = 0; kw < KHW; ++kw)
                        {
                            unsigned iw = ow + kw;

                            if (iw >= HW_PAD && (iw - HW_PAD < IHW))
                            {
                                iw -= HW_PAD;
                                c += AData[ioc + iw * IOC + ih * IOC * IHW] *
                                     BData[ioc + kw * IOC + kh * IOC * KHW];
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void baseline_MM(unsigned ArCr, unsigned AcBr, unsigned BcCc, int LDA,
                        int LDB, int LDC, float const* AData,
                        float const* BData, float* CData, int alpha = 0)
{
    for (int arcr = 0; arcr < ArCr; ++arcr)
    {
        for (int bccc = 0; bccc < BcCc; ++bccc)
        {
            if (alpha == 0)
            {
                CData[arcr * LDC + bccc] = 0.f;
            }
            for (int i = 0; i < AcBr; ++i)
            {
                CData[arcr * LDC + bccc] +=
                    AData[arcr * LDA + i] * BData[i * LDB + bccc];
            }
        }
    }
}

inline void baseline_MM(unsigned ArCr, unsigned AcBr, unsigned BcCc, int ARS,
                        int ACS, int BRS, int BCS, int CRS, int CCS,
                        float const* AData, float const* BData, float* CData,
                        int alpha = 0)
{
    for (int arcr = 0; arcr < ArCr; ++arcr)
    {
        for (int bccc = 0; bccc < BcCc; ++bccc)
        {
            if (alpha == 0)
            {
                CData[arcr * CRS + bccc * CCS] = 0.f;
            }
            for (int i = 0; i < AcBr; ++i)
            {
                CData[arcr * CRS + bccc * CCS] +=
                    AData[arcr * ARS + i * ACS] * BData[i * BRS + bccc * BCS];
            }
        }
    }
}

inline void baseline_matrix_bias(unsigned ArCr, unsigned BcCc, int CRS, int CCS,
                                 int bias_RS, int bias_CS, float* CData,
                                 float const* bias)
{
    for (int arcr = 0; arcr < ArCr; ++arcr)
    {
        for (int bccc = 0; bccc < BcCc; ++bccc)
        {
            CData[arcr * CRS + bccc * CCS] +=
                bias[arcr * bias_RS + bccc * bias_CS];
        }
    }
}

inline void baseline_matrix_elementwise_multiply(unsigned ArCr, unsigned BcCc,
                                                 int CRS, int CCS, int other_RS,
                                                 int other_CS, float* CData,
                                                 float const* other)
{
    for (int arcr = 0; arcr < ArCr; ++arcr)
    {
        for (int bccc = 0; bccc < BcCc; ++bccc)
        {
            CData[arcr * CRS + bccc * CCS] *=
                other[arcr * other_RS + bccc * other_CS];
        }
    }
}

template <class PlusOp, class MultipliesOp>
inline void baseline_MM_op_pair(unsigned ArCr, unsigned AcBr, unsigned BcCc,
                                int ARS, int ACS, int BRS, int BCS, int CRS,
                                int CCS, float const* AData, float const* BData,
                                float* CData, int alpha, float identity_value,
                                PlusOp plus_op, MultipliesOp multiplies_op)
{
    for (int arcr = 0; arcr < ArCr; ++arcr)
    {
        for (int bccc = 0; bccc < BcCc; ++bccc)
        {
            if (alpha == 0)
            {
                CData[arcr * CRS + bccc * CCS] = identity_value;
            }
            for (int i = 0; i < AcBr; ++i)
            {
                float mult = multiplies_op(AData[arcr * ARS + i * ACS],
                                           BData[i * BRS + bccc * BCS]);
                CData[arcr * CRS + bccc * CCS] =
                    plus_op(CData[arcr * CRS + bccc * CCS], mult);
            }
        }
    }
}

inline void baseline_MM_row_col_major(unsigned ArCr, unsigned AcBr,
                                      unsigned BcCc, int LDA, int LDB, int LDC,
                                      float const* AData, float const* BData,
                                      float* CData)
{
    for (int arcr = 0; arcr < ArCr; ++arcr)
    {
        for (int bccc = 0; bccc < BcCc; ++bccc)
        {
            CData[arcr * LDC + bccc] = 0.f;
            for (int i = 0; i < AcBr; ++i)
            {
                CData[arcr * LDC + bccc] +=
                    AData[arcr * LDA + i] * BData[i + bccc * LDB];
            }
        }
    }
}

inline void baseline_Conv(unsigned COUT, unsigned CIN, unsigned OH, int OW,
                          int KH, int KW, float const* AData,
                          float const* BData, float* CData)
{
    // int IH = OH + KH - 1;
    int IW = OW + KW - 1;
    for (int cout = 0; cout < COUT; ++cout)
    {
        for (int oh = 0; oh < OH; ++oh)
        {
            for (int ow = 0; ow < OW; ++ow)
            {
                CData[cout + ow * COUT + oh * COUT * OW] = 0.f;
                for (int cin = 0; cin < CIN; ++cin)
                {
                    for (int kh = 0; kh < KH; ++kh)
                    {
                        for (int kw = 0; kw < KW; ++kw)
                        {
                            CData[cout + ow * COUT + oh * COUT * OW] +=
                                AData[cin + (oh + kh) * CIN * IW +
                                      (ow + kw) * CIN] *
                                BData[cout + cin * COUT + kw * CIN * COUT +
                                      kh * CIN * COUT * KW];
                        }
                    }
                }
            }
        }
    }
}

inline void baseline_padded_Conv(unsigned COUT, unsigned CIN, unsigned OH,
                                 int OW, int KH, int KW, int PH, int PW,
                                 float const* AData, float const* BData,
                                 float* CData)
{
    int IH = OH + KH - 1 - 2 * PH;
    int IW = OW + KW - 1 - 2 * PW;

    for (int cout = 0; cout < COUT; ++cout)
    {
        for (int oh = 0; oh < OH; ++oh)
        {
            for (int ow = 0; ow < OW; ++ow)
            {
                CData[cout + ow * COUT + oh * COUT * OW] = 0.f;
                for (int cin = 0; cin < CIN; ++cin)
                {
                    for (int kh = 0; kh < KH; ++kh)
                    {
                        int ih = oh + kh;

                        if (ih >= PH && (ih - PH < IH))
                        {
                            ih -= PH;

                            for (int kw = 0; kw < KW; ++kw)
                            {
                                int iw = ow + kw;

                                if (iw >= PW && (iw - PW < IW))
                                {
                                    iw -= PW;

                                    CData[cout + ow * COUT + oh * COUT * OW] +=
                                        AData[cin + ih * CIN * IW + iw * CIN] *
                                        BData[cout + cin * COUT +
                                              kw * CIN * COUT +
                                              kh * CIN * COUT * KW];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void baseline_3DConv(int OX, int OY, int OZ, int KX, int KY, int KZ,
                            float const* AData, float const* BData,
                            float* CData)
{
    // int IX = OX + KX - 1;
    int IY = OY + KY - 1;
    int IZ = OZ + KZ - 1;

    for (int ox = 0; ox < OX; ++ox)
    {
        for (int oy = 0; oy < OY; ++oy)
        {
            for (int oz = 0; oz < OZ; ++oz)
            {
                CData[ox * OY * OZ + oy * OZ + oz] = 0.f;
                for (int kx = 0; kx < KX; ++kx)
                {
                    for (int ky = 0; ky < KY; ++ky)
                    {
                        for (int kz = 0; kz < KZ; ++kz)
                        {
                            CData[ox * OY * OZ + oy * OZ + oz] +=
                                AData[(ox + kx) * IY * IZ + (oy + ky) * IZ +
                                      (oz + kz)] *
                                BData[kx * KY * KZ + ky * KZ + kz];
                        }
                    }
                }
            }
        }
    }
}

inline void baseline_Conv_NCHW8c(unsigned GOUT, unsigned COUT, unsigned GIN,
                                 unsigned CIN, unsigned OH, int OW, int KH,
                                 int KW, float const* AData, float const* BData,
                                 float* CData)
{
    int IH = OH + KH - 1;
    int IW = OW + KW - 1;
    for (int gout = 0; gout < GOUT; ++gout)
    {
        for (int cout = 0; cout < COUT; ++cout)
        {
            for (int oh = 0; oh < OH; ++oh)
            {
                for (int ow = 0; ow < OW; ++ow)
                {
                    // C[gout][h][w][cout]
                    CData[((gout * OH + oh) * OW + ow) * COUT + cout] = 0.f;
                    for (int gin = 0; gin < GIN; ++gin)
                    {
                        for (int cin = 0; cin < CIN; ++cin)
                        {
                            for (int kh = 0; kh < KH; ++kh)
                            {
                                for (int kw = 0; kw < KW; ++kw)
                                {
                                    CData[((gout * OH + oh) * OW + ow) * COUT +
                                          cout] +=
                                        AData[((gin * IH + (oh + kh)) * IW +
                                               (ow + kw)) *
                                                  CIN +
                                              cin] *
                                        // B[gin][gout][cin][kh][kw][cout]
                                        BData[((((gin * GOUT + gout) * CIN +
                                                 cin) *
                                                    KH +
                                                kh) *
                                                   KW +
                                               kw) *
                                                  COUT +
                                              cout];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void baseline_Conv_NCHW8c_multiplies_max(unsigned GOUT, unsigned COUT,
                                                unsigned GIN, unsigned CIN,
                                                unsigned OH, int OW, int KH,
                                                int KW, float const* AData,
                                                float const* BData,
                                                float*       CData)
{
    int IH = OH + KH - 1;
    int IW = OW + KW - 1;
    for (int gout = 0; gout < GOUT; ++gout)
    {
        for (int cout = 0; cout < COUT; ++cout)
        {
            for (int oh = 0; oh < OH; ++oh)
            {
                for (int ow = 0; ow < OW; ++ow)
                {
                    // C[gout][h][w][cout]
                    CData[((gout * OH + oh) * OW + ow) * COUT + cout] =
                        -std::numeric_limits<float>::infinity();
                    for (int gin = 0; gin < GIN; ++gin)
                    {
                        for (int cin = 0; cin < CIN; ++cin)
                        {
                            for (int kh = 0; kh < KH; ++kh)
                            {
                                for (int kw = 0; kw < KW; ++kw)
                                {
                                    float temp =
                                        AData[((gin * IH + (oh + kh)) * IW +
                                               (ow + kw)) *
                                                  CIN +
                                              cin] *
                                        // B[gin][gout][cin][kh][kw][cout]
                                        BData[((((gin * GOUT + gout) * CIN +
                                                 cin) *
                                                    KH +
                                                kh) *
                                                   KW +
                                               kw) *
                                                  COUT +
                                              cout];

                                    float curr =
                                        CData[((gout * OH + oh) * OW + ow) *
                                                  COUT +
                                              cout];

                                    CData[((gout * OH + oh) * OW + ow) * COUT +
                                          cout] = temp > curr ? temp : curr;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
