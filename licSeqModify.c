//
// modify by szg on 2022/3/20.
//
//  Mult process of LIC Visualization

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#define	 DISCRETE_FILTER_SIZE	1024
#define  LOWPASS_FILTR_LENGTH	8.00000f
#define	 LINE_SQUARE_CLIP_MAX	100000.0f
#define	 VECTOR_COMPONENT_MIN   0.050000f
#pragma warning(disable:4996)

void     ReadVector(int xres, int yres, double* pVectr, int* pVectrFlag);
void	 NormalizVectrs(int  n_xres, int     n_yres, double* pVectr);
void     GenBoxFiltrLUT(int  LUTsiz, double* p_LUT0, double* p_LUT1);
void     MakeWhiteNoise(int  n_xres, int     n_yres, unsigned char* pNoise);
void	 FlowImagingLIC(int  n_xres, int     n_yres, double* pVectr, unsigned char* pNoise,
    unsigned char* pImage, double* p_LUT0, double* p_LUT1, double  krnlen, int myid, int numprocs, int* pVectrFlag);
void 	 WriteImage2PPM(int  n_xres, int     n_yres, int* pVectrFlag, unsigned char* pImage, char* f_name);


int	main(int argc, char** argv)
{
    //定义MPI_Status类型变量，status记录MPI_Recv里接受TotalpImage的状态
    //定义int类型变量，myid表示进程号，numprocs表示进程数，root用于mpi广播函数判断主进程
    //定义double类型变量end，end1用于记录时间点
    double  start, end;
    MPI_Status status;
    int  myid, numprocs = 0, i;
    int root = 0;
    int left = 110, right = 140, low = 12, high = 32;
    float res = 0.25;
    int				n_xres = (right - left) / res + 1;
    int				n_yres = (high - low) / res + 1;

    double* pVectr = (double*)malloc(sizeof(double) * n_xres * n_yres * 2);//数据集pVector数据集小数为7位，而float只能表示6位，数据失真，故修改为double类型
    int* pVectrFlag = (int*)malloc(sizeof(int) * n_xres * n_yres * 2);
    double* p_LUT0 = (double*)malloc(sizeof(double) * DISCRETE_FILTER_SIZE);
    double* p_LUT1 = (double*)malloc(sizeof(double) * DISCRETE_FILTER_SIZE);
    unsigned char* pNoise = (unsigned char*)malloc(sizeof(unsigned char) * n_xres * n_yres);
    unsigned char* pImage = (unsigned char*)malloc(sizeof(unsigned char) * n_xres * n_yres);
    MPI_Init(&argc, &argv);                            //并行区,初始化
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);              //获得进程id
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);          //获得进程个数
    if (myid != 0)                                     //子进程入口
    {
        MPI_Bcast(pVectr, n_xres * n_yres * 2, MPI_DOUBLE, root, MPI_COMM_WORLD);             //各个分进程广播接收主进程发来的pVector
        MPI_Bcast(pNoise, n_xres * n_yres, MPI_CHAR, root, MPI_COMM_WORLD);                  //各个分进程广播接收主进程发来的pNoise
        MPI_Bcast(p_LUT0, DISCRETE_FILTER_SIZE, MPI_DOUBLE, root, MPI_COMM_WORLD);            //各个分进程广播接收主进程发来的p_LUT0
        MPI_Bcast(p_LUT1, DISCRETE_FILTER_SIZE, MPI_DOUBLE, root, MPI_COMM_WORLD);            //各个分进程广播接收主进程发来的p_LUT1
        FlowImagingLIC(n_xres, n_yres, pVectr, pNoise, pImage, p_LUT0, p_LUT1, 
            LOWPASS_FILTR_LENGTH, myid - 1, numprocs, pVectrFlag);                           //各个分进程计算各自区域的pImage 
        MPI_Send(pImage, n_xres * n_yres, MPI_CHAR, 0, 99, MPI_COMM_WORLD);                  //各个分进程发送各自计算的pImage
    }
    else                                                                                                //主进程入口
    {
        start = MPI_Wtime();                                                                           //记录开始时间点
        unsigned char* TotalpImage = (unsigned char*)malloc(sizeof(unsigned char) * n_xres * n_yres);  //定义TotalpImage用于存储各个进程的数据
        ReadVector(n_xres, n_yres, pVectr, pVectrFlag);                                
        NormalizVectrs(n_xres, n_yres, pVectr);  
        MPI_Bcast(pVectr, n_xres * n_yres * 2, MPI_DOUBLE, root, MPI_COMM_WORLD);                       //主进程广播pVector到分进程    
        MakeWhiteNoise(n_xres, n_yres, pNoise);      
        MPI_Bcast(pNoise, n_xres * n_yres, MPI_CHAR, root, MPI_COMM_WORLD);                            //主进程广播pNoise到分进程   
        GenBoxFiltrLUT(DISCRETE_FILTER_SIZE, p_LUT0, p_LUT1);       
        MPI_Bcast(p_LUT0, DISCRETE_FILTER_SIZE, MPI_DOUBLE, root, MPI_COMM_WORLD);                      //主进程广播p_LUT0到分进程   
        MPI_Bcast(p_LUT1, DISCRETE_FILTER_SIZE, MPI_DOUBLE, root, MPI_COMM_WORLD);                      //主进程广播p_LUT1到分进程
        for (i = 1;i < numprocs; i++)                                                                  //循环numprocs-1次用于接收除主进程之外的进程的值
        {
            MPI_Recv(pImage, n_xres * n_yres, MPI_CHAR, i, 99, MPI_COMM_WORLD, &status);               //接受各个分进程的pImage
            int k = i - 1;                                                                             //初值
            for (int j = 0; j < n_yres; j++)                                                           //列，顺序计算
            {
                for (; k < n_xres; k += numprocs-1)                                                    //行，每numprocs-1个单元，接收一个piamge[i]
                {
                    int		index = (j * n_xres + k);                                                  //定位对应数组地址
                    TotalpImage[index] = pImage[index];                                                //将分进程的数值赋值到总进程里                         
                }
                k = k - n_xres;                                                                        //行末，冗余部分加到下一行
            }
        }                                                                           //记录结束时间点
        WriteImage2PPM(n_xres, n_yres, pVectrFlag, TotalpImage, "LIC.ppm");
        end = MPI_Wtime();
        printf("time=%f\n", (end - start));
    }
    MPI_Finalize();
    free(pVectr);	pVectr = NULL;
    free(p_LUT0);	p_LUT0 = NULL;
    free(p_LUT1);	p_LUT1 = NULL;
    free(pNoise);	pNoise = NULL;
    free(pImage);	pImage = NULL;
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
void ReadVector(int xres, int yres, double* pVectr, int* pVectrFlag) {
    FILE* fp;
    if ((fp = fopen("E:\\Data\\fielddata.txt", "r")) == NULL) {
        printf("error in reading file !\n");
        exit(1);
    }
    double f1, f2, f3, f4;
    int index = 0;
    while (!feof(fp)) {
        if (fscanf(fp, "%lf %lf %lf %lf", &f1, &f2, &f3, &f4) == EOF)
            break;
        //printf( "%f %f %f %f \n", f1, f2, f3, f4);
        pVectr[index] = f3;
        if ((int)(f3 - 9999) == 0) {
            pVectrFlag[index] = 1;
        }
        index++;

        pVectr[index] = f4;
        if ((int)(f4 - 9999) == 0) {
            pVectrFlag[index] = 1;
        }
        index++;
    }
    fclose(fp);

}


///		normalize the vector field     ///
void    NormalizVectrs(int  n_xres, int  n_yres, double* pVectr)
{
    for (int j = 0; j < n_yres; j++)
        for (int i = 0; i < n_xres; i++)
        {
            int		index = (j * n_xres + i) << 1;
            double	vcMag = (double)(sqrt((double)(pVectr[index] * pVectr[index] + pVectr[index + 1] * pVectr[index + 1])));//改为double型

            double	scale = (vcMag == 0.0f) ? 0.0f : 1.0f / vcMag;                                                          //改为double型
            pVectr[index] *= scale;
            pVectr[index + 1] *= scale;

        }

}


///		make white noise as the LIC input texture     ///
void	MakeWhiteNoise(int  n_xres, int  n_yres, unsigned char* pNoise)
{
    for (int j = 0; j < n_yres; j++)
        for (int i = 0; i < n_xres; i++)
        {
            int  r = rand();
            r = ((r & 0xff) + ((r & 0xff00) >> 8)) & 0xff;
            pNoise[j * n_xres + i] = (unsigned char)r;
        }
}


///		generate box filter LUTs     ///
void    GenBoxFiltrLUT(int  LUTsiz, double* p_LUT0, double* p_LUT1)
{
    for (int i = 0; i < LUTsiz; i++)  p_LUT0[i] = p_LUT1[i] = i;
}


///		write the LIC image to a PPM file     ///
void	WriteImage2PPM(int  n_xres, int  n_yres, int* pVectrFlag, unsigned char* pImage, char* f_name)
{
    FILE* o_file;
    if ((o_file = fopen(f_name, "w")) == NULL)
    {
        printf("Can't open output file\n");
        return;
    }

    fprintf(o_file, "P6\n%d %d\n255\n", n_xres, n_yres);

    for (int j = n_yres - 1; j > -1; j--)
        for (int i = 0; i < n_xres; i++)
        {
            unsigned  char	unchar = pImage[j * n_xres + i];
            ///leave the land pixel untouched
            if (pVectrFlag[(j * n_xres + i) * 2] == 1) {
                unchar = (unsigned char)255;
                //printf("%d %d \n", i, j);
            }
            //printf("%d %d %d\n", i, j, unchar);
            fprintf(o_file, "%c%c%c", unchar, unchar, unchar);
        }

    fclose(o_file);	o_file = NULL;
}


		///flow imaging (visualization) through Line Integral Convolution     ///
void	FlowImagingLIC(int     n_xres, int     n_yres, double* pVectr, unsigned char* pNoise, unsigned char* pImage,
    double* p_LUT0, double* p_LUT1, double   krnlen, int myid, int numprocs, int* pVectrFlag) //引入myid，numprocs，pVectrFlag参数
{
    int		vec_id;						///ID in the VECtor buffer (for the input flow field)
    int		advDir;						///ADVection DIRection (0: positive;  1: negative)
    int		advcts;						///number of ADVeCTion stepS per direction (a step counter)
    int		ADVCTS = (int)(krnlen * 3);	///MAXIMUM number of advection steps per direction to break dead loops

    //修改为double型//
    double	vctr_x;						///x-component  of the VeCToR at the forefront point 
    double	vctr_y;						///y-component  of the VeCToR at the forefront point 
    double	clp0_x;						///x-coordinate of CLiP point 0 (current)
    double	clp0_y;						///y-coordinate of CLiP point 0	(current)
    double	clp1_x;						///x-coordinate of CLiP point 1 (next   )
    double	clp1_y;						///y-coordinate of CLiP point 1 (next   )
    double	samp_x;						///x-coordinate of the SAMPle in the current pixel
    double	samp_y;						///y-coordinate of the SAMPle in the current pixel
    double	tmpLen;						///TeMPorary LENgth of a trial clipped-segment
    double	segLen;						///SEGment   LENgth
    double	curLen;						///CURrent   LENgth of the streamline
    double	prvLen;						///PReVious  LENgth of the streamline
    double	W_ACUM;						///ACcuMulated Weight from the seed to the current streamline forefront
    double	texVal;						///TEXture VALue
    double	smpWgt;						///WeiGhT of the current SaMPle
    double	t_acum[2];					///two ACcUMulated composite Textures for the two directions, perspectively
    double	w_acum[2];					///two ACcUMulated Weighting values   for the two directions, perspectively
    double* wgtLUT = NULL;				///WeiGhT Look Up Table pointing to the target filter LUT
    double	len2ID = (DISCRETE_FILTER_SIZE - 1) / krnlen;	///map a curve LENgth TO an ID in the LUT

    ///for each pixel in the 2D output LIC image///
    int i = myid;                                    //根据不同的进程初值不同
    for (int j = 0; j < n_yres; j++)                 //列，顺序进行
    {
        for (; i < n_xres; i += numprocs - 1)        //行，每numprocs-1个单元计算一个值插入
        {
            ///init the composite texture accumulators and the weight accumulators///
            t_acum[0] = t_acum[1] = w_acum[0] = w_acum[1] = 0.0;

            ///for either advection direction///
            for (advDir = 0; advDir < 2; advDir++)
            {
                ///init the step counter, curve-length measurer, and streamline seed///
                advcts = 0;
                curLen = 0.0f;
                clp0_x = i + 0.5f;
                clp0_y = j + 0.5f;

                ///access the target filter LUT///
                wgtLUT = (advDir == 0) ? p_LUT0 : p_LUT1;

                ///until the streamline is advected long enough or a tightly  spiralling center / focus is encountered///
                while (curLen < krnlen && advcts < ADVCTS)
                {
                    ///access the vector at the sample///
                    vec_id = ((int)(clp0_y)*n_xres + (int)(clp0_x)) << 1;
                    vctr_x = pVectr[vec_id];
                    vctr_y = pVectr[vec_id + 1];
                    //把源代码部分删掉，因为数据没有等于0的时候，其实没用。  
                    ///in case of a critical point///
                    //if (vctr_x == 0.0f && vctr_y == 0.0f)
                    //{
                    //    t_acum[advDir] = (advcts == 0) ? 0.0f : t_acum[advDir];		   ///this line is indeed unnecessary
                    //    w_acum[advDir] = (advcts == 0) ? 1.0f : w_acum[advDir];
                    //    break;
                    //}                
                    //就是陆地的矢量没必要传进LIC里面计算，需要把他们滤掉   
                    //结果正确是因为在写文件函数时又把陆地部分赋值为白色了，所以这部分积分没体现出来，属于浪费计算资源了。
                    if (pVectrFlag[vec_id] == 1 && pVectrFlag[vec_id + 1] == 1) {    //遇到陆地部分跳出循环
                        break;
                    }
                    ///negate the vector for the backward-advection case///
                    vctr_x = (advDir == 0) ? vctr_x : -vctr_x;
                    vctr_y = (advDir == 0) ? vctr_y : -vctr_y;

                    ///clip the segment against the pixel boundaries --- find the shorter from the two clipped segments///
                    ///replace  all  if-statements  whenever  possible  as  they  might  affect the computational speed///
                    segLen = LINE_SQUARE_CLIP_MAX;
                    segLen = (vctr_x < -VECTOR_COMPONENT_MIN) ? ((int)(clp0_x)-clp0_x) / vctr_x : segLen;
                    segLen = (vctr_x > VECTOR_COMPONENT_MIN) ? ((int)((int)(clp0_x)+1.5f) - clp0_x) / vctr_x : segLen;
                    segLen = (vctr_y < -VECTOR_COMPONENT_MIN) ? (((tmpLen = ((int)(clp0_y)-clp0_y) / vctr_y) < segLen) ? tmpLen : segLen) : segLen;
                    segLen = (vctr_y > VECTOR_COMPONENT_MIN) ? (((tmpLen = ((int)((int)(clp0_y)+1.5f) - clp0_y) / vctr_y) < segLen) ? tmpLen : segLen) : segLen;

                    ///update the curve-length measurers///
                    prvLen = curLen;
                    curLen += segLen;
                    segLen += 0.0004f;

                    ///check if the filter has reached either end///
                    segLen = (curLen > krnlen) ? ((curLen = krnlen) - prvLen) : segLen;

                    ///obtain the next clip point///
                    clp1_x = clp0_x + vctr_x * segLen;
                    clp1_y = clp0_y + vctr_y * segLen;

                    ///obtain the middle point of the segment as the texture-contributing sample///
                    samp_x = (clp0_x + clp1_x) * 0.5f;
                    samp_y = (clp0_y + clp1_y) * 0.5f;

                    ///obtain the texture value of the sample///
                    texVal = pNoise[(int)(samp_y)*n_xres + (int)(samp_x)];

                    ///update the accumulated weight and the accumulated composite texture (texture x weight)///
                    W_ACUM = wgtLUT[(int)(curLen * len2ID)];
                    smpWgt = W_ACUM - w_acum[advDir];
                    w_acum[advDir] = W_ACUM;
                    t_acum[advDir] += texVal * smpWgt;

                    ///update the step counter and the "current" clip point///
                    advcts++;
                    clp0_x = clp1_x;
                    clp0_y = clp1_y;

                    ///check if the streamline has gone beyond the flow field///
                    if (clp0_x < 0.0f || clp0_x >= n_xres || clp0_y < 0.0f || clp0_y >= n_yres)  break;
                }
            }

            ///normalize the accumulated composite texture///
            texVal = (t_acum[0] + t_acum[1]) / (w_acum[0] + w_acum[1]);

            ///clamp the texture value against the displayable intensity range [0, 255]
            texVal = (texVal < 0.0f) ? 0.0f : texVal;
            texVal = (texVal > 255.0f) ? 255.0f : texVal;
            pImage[j * n_xres + i] = (unsigned char)texVal;
        }
        i = i - n_xres;                              //行末冗余部分，到下行继续冗余部分处开始计算
    }
}



