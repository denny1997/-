#include <stdio.h>
#include <iostream>
#include <math.h>
using namespace std;
const int max_movie = 1683;
const int max_user = 944;
#define THREAD_NUM 256
#define BLOCK_NUM 32

__global__ static void test(float* rate,float* result)
{
	const int tid=threadIdx.x;
	const int bid=blockIdx.x;
	int i,k,j;
	float sum;
	for(i=bid*THREAD_NUM+tid;i<max_movie-1;i+=BLOCK_NUM*THREAD_NUM)
	{
		for(k=1;k<max_movie;k++)
		{
			if(k==i+1)
				continue;
			else
			{
				sum=0;
				//count=0;
				for(j=1;j<max_user;j++)
				{
					if((rate[j*max_movie+i+1]==-1)||(rate[j*max_movie+k]==-1))
						continue;
					else
					{
						sum=sum+(rate[j*max_movie+i+1]-rate[j*max_movie+k])*(rate[j*max_movie+i+1]-rate[j*max_movie+k]);
						//count++;
					}
				}
				result[(i+1)*max_movie+k]=sqrt(sum);
			}
		}


	}

}


int main()
{
	cout<<"start:"<<endl;
	FILE *fp = fopen("/home/3160102482/myhelloworld/ml-100k/u2.base","r");

	float a[max_movie*max_user];
	float* res;
	res=(float*)malloc(sizeof(float)*max_movie*max_movie);
	int movieid,userid,stamp,i,j;
	float rating;
	float* gpudata;
	float* result;
	float min;
	int minindex1,minindex2,minindex3,k;
	//cout<<"yes"<<endl;
	//cout<<"4";
	for(i=0;i<max_user;i++)
		for(j=0;j<max_movie;j++)
			a[i*max_movie+j]=-1;
	//cout<<"3";
	cudaMalloc((void**)&gpudata,sizeof(float)*max_movie*max_user);
	cudaMalloc((void**)&result,sizeof(float)*max_movie*max_movie);
	while(!feof(fp)){
		fscanf(fp,"%d%d%f%d",&userid, &movieid, &rating, &stamp);
		a[userid*max_movie+movieid]=rating;
	}
	fclose(fp);
	cudaMemcpy(gpudata,a,sizeof(float)*max_movie*max_user,cudaMemcpyHostToDevice);
	//cout<<"1";
	test<<<BLOCK_NUM,THREAD_NUM,0>>>(gpudata,result);
	cudaMemcpy(res,result,sizeof(float)*max_movie*max_movie,cudaMemcpyDeviceToHost);

	cudaFree(gpudata);
	cudaFree(result);
	fp = fopen("/home/3160102482/myhelloworld/ml-100k/u2.test","r");
	while(!feof(fp)){
			fscanf(fp,"%d%d%f%d",&userid, &movieid, &rating, &stamp);
			min=100;
			for(i=movieid*max_movie+1;i<(movieid+1)*max_movie;i++)
			{
				if(i==(movieid*max_movie+movieid))
					continue;
				else
				{
					k=i-movieid*max_movie;
					if((a[userid*max_movie+k]!=-1)&&(res[i]<min))
					{
						minindex1=k;
						min=res[i];
					}

				}
			}
			min=100;
			for(i=movieid*max_movie+1;i<(movieid+1)*max_movie;i++)
			{
				if((i==(movieid*max_movie+movieid))||(i==(movieid*max_movie+minindex1)))
					continue;
				else
				{
					k=i-movieid*max_movie;
					if((a[userid*max_movie+k]!=-1)&&(res[i]<min))
					{
						minindex2=k;
						min=res[i];
					}

				}
			}
			min=100;
			for(i=movieid*max_movie+1;i<(movieid+1)*max_movie;i++)
			{
				if((i==(movieid*max_movie+movieid))||(i==(movieid*max_movie+minindex1))||(i==(movieid*max_movie+minindex2)))
					continue;
				else
				{
					k=i-movieid*max_movie;
					if((a[userid*max_movie+k]!=-1)&&(res[i]<min))
					{
						minindex3=k;
						min=res[i];
					}

				}
			}
			cout<<"userid:"<<userid<<" ";
			cout<<"movieid:"<<movieid<<" "; 
			cout<<"estimate:"<<(float)(a[userid*max_movie+minindex1]+a[userid*max_movie+minindex2]+a[userid*max_movie+minindex3])/3<<" ";
			cout<<"real:"<<rating<<endl;
			cout<<endl;
		}
		fclose(fp);
	//cout<<res[1687];
}
