//
// Created by 洪锋 on 2022/1/27.
//
//  single process of LIC Visualization

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define	 DISCRETE_FILTER_SIZE	1024
#define  LOWPASS_FILTR_LENGTH	8.00000f
#define	 LINE_SQUARE_CLIP_MAX	100000.0f
#define	 VECTOR_COMPONENT_MIN   0.050000f


void     ReadVector(int xres, int yres, float*  pVectr, int* pVectrFlag);
void	 NormalizVectrs(int  n_xres,  int     n_yres,  float*   pVectr);
void     GenBoxFiltrLUT(int  LUTsiz,  float*  p_LUT0,  float*   p_LUT1);
void     MakeWhiteNoise(int  n_xres,  int     n_yres,  unsigned char*  pNoise);
void	 FlowImagingLIC(int  n_xres,  int     n_yres,  float*   pVectr,   unsigned char*  pNoise,
                        unsigned char*  pImage,  float*  p_LUT0,  float*  p_LUT1,  float  krnlen);
void 	 WriteImage2PPM(int  n_xres,  int     n_yres,  int* pVectrFlag,   unsigned char*  pImage,     char*  f_name);


int	main(int argc, char **argv)
{
    clock_t start,end;
    start = clock();

    int left = 110, right = 140, low =12, high = 32;
    float res=0.25;
    int				n_xres = (right-left)/res+1;
    int				n_yres = (high-low)/res+1;

    float*			pVectr = (float*         ) malloc( sizeof(float        ) * n_xres * n_yres * 2 );
    int*			pVectrFlag = (int*       ) malloc( sizeof(int          ) * n_xres * n_yres * 2 );
    float*			p_LUT0 = (float*		 ) malloc( sizeof(float        ) * DISCRETE_FILTER_SIZE);
    float*			p_LUT1 = (float*		 ) malloc( sizeof(float        ) * DISCRETE_FILTER_SIZE);
    unsigned char*	pNoise = (unsigned char* ) malloc( sizeof(unsigned char) * n_xres * n_yres     );
    unsigned char*	pImage = (unsigned char* ) malloc( sizeof(unsigned char) * n_xres * n_yres     );

    ReadVector(n_xres, n_yres, pVectr, pVectrFlag);
    NormalizVectrs(n_xres, n_yres, pVectr);
    MakeWhiteNoise(n_xres, n_yres, pNoise);
    GenBoxFiltrLUT(DISCRETE_FILTER_SIZE, p_LUT0, p_LUT1);
    FlowImagingLIC(n_xres, n_yres, pVectr, pNoise, pImage, p_LUT0, p_LUT1, LOWPASS_FILTR_LENGTH);
    WriteImage2PPM(n_xres, n_yres, pVectrFlag, pImage, "LIC.ppm");

    free(pVectr);	pVectr = NULL;
    free(p_LUT0);	p_LUT0 = NULL;
    free(p_LUT1);	p_LUT1 = NULL;
    free(pNoise);	pNoise = NULL;
    free(pImage);	pImage = NULL;

    end = clock();
    printf("time=%f\n",(double)(end-start)/1000000);
    return 0;
}


///		synthesize a saddle-shaped vector field     ///
//void	SyntheszSaddle(int  n_xres,  int  n_yres,  float*  pVectr)
//{
//    for(int  j = 0;  j < n_yres;  j ++)
//        for(int  i = 0;  i < n_xres;  i ++)
//        {
//            int	 index = (  (n_yres - 1 - j) * n_xres + i  )  <<  1;
//            pVectr[index    ] = - ( j / (n_yres - 1.0f) - 0.5f );
//            pVectr[index + 1] =     i / (n_xres - 1.0f) - 0.5f;
//        }
//}


///		read the vector field     ///
void ReadVector(int xres, int yres, float*  pVectr, int* pVectrFlag) {
    FILE *fp;
    if ((fp = fopen("fielddata.txt", "r")) == NULL) {
        printf("error in reading file !\n");
        exit(1);
    }
    float f1, f2, f3, f4;
    int index = 0;
    while (!feof(fp)) {
        if (fscanf(fp, "%f %f %f %f", &f1, &f2, &f3, &f4) == EOF)
            break;
        //printf( "%f %f %f %f \n", f1, f2, f3, f4);
        pVectr[index] = f3;
        if((int)(f3-9999)==0){
            pVectrFlag[index] = 1;
        }
        index++;

        pVectr[index] = f4;
        if((int)(f4-9999)==0){
            pVectrFlag[index] = 1;
        }
        index++;
    }
    fclose(fp);

}


///		normalize the vector field     ///
void    NormalizVectrs(int  n_xres,  int  n_yres,  float*  pVectr)
{
    for(int	 j = 0;  j < n_yres;  j ++)
        for(int	 i = 0;  i < n_xres;  i ++)
        {
            int		index = (j * n_xres + i) << 1;
            float	vcMag = (float)(  sqrt( (double)(pVectr[index] * pVectr[index] + pVectr[index + 1] * pVectr[index + 1]) )  );

            float	scale = (vcMag == 0.0f) ? 0.0f : 1.0f / vcMag;
            pVectr[index    ] *= scale;
            pVectr[index + 1] *= scale;
        }
}


///		make white noise as the LIC input texture     ///
void	MakeWhiteNoise(int  n_xres,  int  n_yres,  unsigned char*  pNoise)
{
    for(int  j = 0;   j < n_yres;  j ++)
        for(int  i = 0;   i < n_xres;  i ++)
        {
            int  r = rand();
            r = (  (r & 0xff) + ( (r & 0xff00) >> 8 )  ) & 0xff;
            pNoise[j * n_xres + i] = (unsigned char) r;
        }
}


///		generate box filter LUTs     ///
void    GenBoxFiltrLUT(int  LUTsiz,  float*  p_LUT0,  float*  p_LUT1)
{
    for(int  i = 0;  i < LUTsiz;  i ++)  p_LUT0[i] = p_LUT1[i] = i;
}


///		write the LIC image to a PPM file     ///
void	WriteImage2PPM(int  n_xres,  int  n_yres,   int* pVectrFlag, unsigned char*  pImage,  char*  f_name)
{
    FILE*	o_file;
    if(   ( o_file = fopen(f_name, "w") )  ==  NULL   )
    {
        printf("Can't open output file\n");
        return;
    }

    fprintf(o_file, "P6\n%d %d\n255\n", n_xres, n_yres);

    for(int  j = n_yres-1;  j > -1 ;  j--)
        for(int  i = 0;  i < n_xres;  i ++)
        {
            unsigned  char	unchar = pImage[j * n_xres + i];
            ///leave the land pixel untouched
            if (pVectrFlag[(j * n_xres + i)*2]  == 1 ){
                unchar = (unsigned char) 255;
                //printf("%d %d \n", i, j);
            }
            //printf("%d %d %d\n", i, j, unchar);
            fprintf(o_file, "%c%c%c", unchar, unchar, unchar);
        }

    fclose (o_file);	o_file = NULL;
}


///		flow imaging (visualization) through Line Integral Convolution     ///
void	FlowImagingLIC(int     n_xres,  int     n_yres,  float*  pVectr,  unsigned char*  pNoise,  unsigned char*  pImage,
                       float*  p_LUT0,  float*  p_LUT1,  float   krnlen)
{
    int		vec_id;						///ID in the VECtor buffer (for the input flow field)
    int		advDir;						///ADVection DIRection (0: positive;  1: negative)
    int		advcts;						///number of ADVeCTion stepS per direction (a step counter)
    int		ADVCTS = (int)(krnlen * 3);	///MAXIMUM number of advection steps per direction to break dead loops

    float	vctr_x;						///x-component  of the VeCToR at the forefront point
    float	vctr_y;						///y-component  of the VeCToR at the forefront point
    float	clp0_x;						///x-coordinate of CLiP point 0 (current)
    float	clp0_y;						///y-coordinate of CLiP point 0	(current)
    float	clp1_x;						///x-coordinate of CLiP point 1 (next   )
    float	clp1_y;						///y-coordinate of CLiP point 1 (next   )
    float	samp_x;						///x-coordinate of the SAMPle in the current pixel
    float	samp_y;						///y-coordinate of the SAMPle in the current pixel
    float	tmpLen;						///TeMPorary LENgth of a trial clipped-segment
    float	segLen;						///SEGment   LENgth
    float	curLen;						///CURrent   LENgth of the streamline
    float	prvLen;						///PReVious  LENgth of the streamline
    float	W_ACUM;						///ACcuMulated Weight from the seed to the current streamline forefront
    float	texVal;						///TEXture VALue
    float	smpWgt;						///WeiGhT of the current SaMPle
    float	t_acum[2];					///two ACcUMulated composite Textures for the two directions, perspectively
    float	w_acum[2];					///two ACcUMulated Weighting values   for the two directions, perspectively
    float*	wgtLUT = NULL;				///WeiGhT Look Up Table pointing to the target filter LUT
    float	len2ID = (DISCRETE_FILTER_SIZE - 1) / krnlen;	///map a curve LENgth TO an ID in the LUT

    ///for each pixel in the 2D output LIC image///
    for(int  j = 0;	 j < n_yres;  j ++)
        for(int  i = 0;	 i < n_xres;  i ++)
        {
            ///init the composite texture accumulators and the weight accumulators///
            t_acum[0] = t_acum[1] = w_acum[0] = w_acum[1] = 0.0f;

            ///for either advection direction///
            for(advDir = 0;  advDir < 2;  advDir ++)
            {
                ///init the step counter, curve-length measurer, and streamline seed///
                advcts = 0;
                curLen = 0.0f;
                clp0_x = i + 0.5f;
                clp0_y = j + 0.5f;

                ///access the target filter LUT///
                wgtLUT = (advDir == 0) ? p_LUT0 : p_LUT1;

                ///until the streamline is advected long enough or a tightly  spiralling center / focus is encountered///
                while( curLen < krnlen && advcts < ADVCTS )
                {
                    ///access the vector at the sample///
                    vec_id = ( (int)(clp0_y) * n_xres + (int)(clp0_x) )<<1;
                    vctr_x = pVectr[vec_id    ];
                    vctr_y = pVectr[vec_id + 1];

                    ///in case of a critical point///
                    if( vctr_x == 0.0f && vctr_y == 0.0f )
                    {
                        t_acum[advDir] = (advcts == 0) ? 0.0f : t_acum[advDir];		   ///this line is indeed unnecessary
                        w_acum[advDir] = (advcts == 0) ? 1.0f : w_acum[advDir];
                        break;
                    }

                    ///negate the vector for the backward-advection case///
                    vctr_x = (advDir == 0) ? vctr_x : -vctr_x;
                    vctr_y = (advDir == 0) ? vctr_y : -vctr_y;

                    ///clip the segment against the pixel boundaries --- find the shorter from the two clipped segments///
                    ///replace  all  if-statements  whenever  possible  as  they  might  affect the computational speed///
                    segLen = LINE_SQUARE_CLIP_MAX;
                    segLen = (vctr_x < -VECTOR_COMPONENT_MIN) ? ( (int)(     clp0_x         ) - clp0_x ) / vctr_x : segLen;
                    segLen = (vctr_x >  VECTOR_COMPONENT_MIN) ? ( (int)( (int)(clp0_x) + 1.5f ) - clp0_x ) / vctr_x : segLen;
                    segLen = (vctr_y < -VECTOR_COMPONENT_MIN) ? (((tmpLen = ((int)(clp0_y)-clp0_y)/vctr_y)<segLen)?tmpLen:segLen ): segLen;
                    segLen = (vctr_y >  VECTOR_COMPONENT_MIN) ? (((tmpLen = ((int)((int)(clp0_y) + 1.5f )-clp0_y)/vctr_y)<segLen ) ? tmpLen : segLen): segLen;

                    ///update the curve-length measurers///
                    prvLen = curLen;
                    curLen+= segLen;
                    segLen+= 0.0004f;

                    ///check if the filter has reached either end///
                    segLen = (curLen > krnlen) ? ( (curLen = krnlen) - prvLen ) : segLen;

                    ///obtain the next clip point///
                    clp1_x = clp0_x + vctr_x * segLen;
                    clp1_y = clp0_y + vctr_y * segLen;

                    ///obtain the middle point of the segment as the texture-contributing sample///
                    samp_x = (clp0_x + clp1_x) * 0.5f;
                    samp_y = (clp0_y + clp1_y) * 0.5f;

                    ///obtain the texture value of the sample///
                    texVal = pNoise[ (int)(samp_y) * n_xres + (int)(samp_x) ];

                    ///update the accumulated weight and the accumulated composite texture (texture x weight)///
                    W_ACUM = wgtLUT[ (int)(curLen * len2ID) ];
                    smpWgt = W_ACUM - w_acum[advDir];
                    w_acum[advDir]  = W_ACUM;
                    t_acum[advDir] += texVal * smpWgt;

                    ///update the step counter and the "current" clip point///
                    advcts ++;
                    clp0_x = clp1_x;
                    clp0_y = clp1_y;

                    ///check if the streamline has gone beyond the flow field///
                    if( clp0_x < 0.0f || clp0_x >= n_xres || clp0_y < 0.0f || clp0_y >= n_yres)  break;
                }
            }

            ///normalize the accumulated composite texture///
            texVal = (t_acum[0] + t_acum[1]) / (w_acum[0] + w_acum[1]);

            ///clamp the texture value against the displayable intensity range [0, 255]
            texVal = (texVal <   0.0f) ?   0.0f : texVal;
            texVal = (texVal > 255.0f) ? 255.0f : texVal;
            pImage[j * n_xres + i] = (unsigned char) texVal;
        }

}
