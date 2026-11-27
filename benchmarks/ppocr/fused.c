#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_ms(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1000.0+ts.tv_nsec/1e6;}

static inline float norm(unsigned char v){return ((float)v/255.0f-0.5f)/0.5f;}

int main(int argc,char**argv){
  if(argc<6){fprintf(stderr,"usage: %s src.raw sw sh dw dh [resize|chw]\n",argv[0]);return 2;}
  const char*path=argv[1];int sw=atoi(argv[2]),sh=atoi(argv[3]),dw=atoi(argv[4]),dh=atoi(argv[5]);
  const char*mode=(argc>6)?argv[6]:"resize";
  size_t nsrc=(size_t)sw*sh*3;
  unsigned char*src=malloc(nsrc);FILE*f=fopen(path,"rb");if(!f){perror("open");return 1;}
  if(fread(src,1,nsrc,f)!=nsrc){fprintf(stderr,"short read\n");return 1;}fclose(f);
  size_t nout=(size_t)dw*dh*3;
  float*out=malloc(nout*sizeof(float));
  double t0=now_ms();
  if(!strcmp(mode,"chw")){
    for(int c=0;c<3;c++)for(int y=0;y<dh;y++)for(int x=0;x<dw;x++)
      out[((size_t)c*dh+y)*dw+x]=norm(src[((size_t)y*dw+x)*3+c]);
  } else {
    double scale=fmin((double)dw/sw,(double)dh/sh);
    int nw=(int)round(sw*scale),nh=(int)round(sh*scale);
    int padx=(dw-nw)/2,pady=(dh-nh)/2;
    float pad=norm(255);
    for(size_t i=0;i<nout;i++)out[i]=pad;
    double rx=(double)sw/nw,ry=(double)sh/nh;
    for(int y=0;y<nh;y++){
      double sy=(y+0.5)*ry-0.5; if(sy<0)sy=0; int y0=(int)sy; double fy=sy-y0; int y1=(y0+1<sh)?y0+1:y0;
      for(int x=0;x<nw;x++){
        double sx=(x+0.5)*rx-0.5; if(sx<0)sx=0; int x0=(int)sx; double fx=sx-x0; int x1=(x0+1<sw)?x0+1:x0;
        const unsigned char*p00=src+((size_t)y0*sw+x0)*3,*p01=src+((size_t)y0*sw+x1)*3;
        const unsigned char*p10=src+((size_t)y1*sw+x0)*3,*p11=src+((size_t)y1*sw+x1)*3;
        for(int c=0;c<3;c++){
          double v=(p00[c]*(1-fx)+p01[c]*fx)*(1-fy)+(p10[c]*(1-fx)+p11[c]*fx)*fy;
          out[((size_t)c*dh+y+pady)*dw+x+padx]=norm((unsigned char)(v<0?0:(v>255?255:v)));
        }
      }
    }
  }
  double t1=now_ms();
  if(argc>7){FILE*g=fopen(argv[7],"wb");fwrite(out,sizeof(float),nout,g);fclose(g);}
  double sum=0;for(size_t i=0;i<nout;i++)sum+=out[i];
  printf("%-8s src=%dx%d dst=%dx%d %.2f ms sum=%.3f\n",mode,sw,sh,dw,dh,t1-t0,sum);
  free(src);free(out);return 0;
}
