#pragma once

void baseline_MM(unsigned ArCr, unsigned AcBr, unsigned BcCc, int LDA, int LDB,
                 int LDC, float const* AData, float const* BData, float* CData,
                 int alpha = 0)
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

void baseline_MM(unsigned ArCr, unsigned AcBr, unsigned BcCc, int ARS, int ACS,
                 int BRS, int BCS, int CRS, int CCS, float const* AData,
                 float const* BData, float* CData, int alpha = 0)
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

void baseline_MM_row_col_major(unsigned ArCr, unsigned AcBr, unsigned BcCc,
                               int LDA, int LDB, int LDC, float const* AData,
                               float const* BData, float* CData)
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

void baseline_Conv(unsigned COUT, unsigned CIN, unsigned OH, int OW, int KH,
                   int KW, float const* AData, float const* BData, float* CData)
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

void baseline_3DConv(int OX, int OY, int OZ, int KX, int KY, int KZ,
                     float const* AData, float const* BData, float* CData)
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

void baseline_Conv_NCHW8c(unsigned GOUT, unsigned COUT, unsigned GIN,
                          unsigned CIN, unsigned OH, int OW, int KH, int KW,
                          float const* AData, float const* BData, float* CData)
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
