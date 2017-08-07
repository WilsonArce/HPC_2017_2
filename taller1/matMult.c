#include <stdio.h>
#include <stdlib.h>

#define N 512

int main (int argc,char **argv)
{
	FILE *f1, *f2, *f3;

	//variables para almacenar las dimensiones de las matrices
	int m1Col, m1Row, m2Col, m2Row;

	//variables para almacenar valores y rerultado
	int a, b, ans;

	f1 = fopen("mat1.txt","r");
	f2 = fopen("mat2.txt","r");
	f3 = fopen("answer.txt","w");
	
	fscanf(f1, "%d", &m1Row); fscanf(f1, "%d", &m1Col);
	fscanf(f2, "%d", &m2Row); fscanf(f2, "%d", &m2Col);

	if (m1Col == m2Row)
	{
		int	**mat1;
		mat1 = (int **)malloc(m1Row * sizeof(int *));

		int	**mat2;
		mat2 = (int **)malloc(m2Row * sizeof(int *));

		for (int i = 0; i < m1Row; i++)
		{
			mat1[i] = (int *)malloc(m1Col * sizeof(int));
			for (int j = 0; j < m1Col; j++)
			{
				fscanf(f1, "%d", &mat1[i][j]);
				getc(f1);
			}
		}

		for (int i = 0; i < m2Row; i++)
		{
			mat2[i] = (int *)malloc(m2Col * sizeof(int));
			for (int j = 0; j < m2Col; j++)
			{
				fscanf(f2, "%d", &mat2[i][j]);
				getc(f2);
			}
		}
		
		fprintf(f3, "%d\n", m1Row);
		fprintf(f3, "%d\n", m2Col);

		int a,b;
		for (int i = 0; i < m1Row; i++) {
	    for (int j = 0; j < m2Col; j++) {
	   		int sum = 0;
	      for (int k = 0; k < m1Col; k++) {
	        a = mat1[i][k];
	        b = mat2[k][j];
	        sum += a * b;
	      }
	      fprintf(f3, "%d,", sum);
	  	}
	  	fseek(f3, -1, SEEK_END);
	  	fprintf(f3, "\n");
	 	}

		fclose(f1); fclose(f2);

		free(mat1); free(mat2);

	}else{
		printf("Las matrices no se puede multiplicar\n");
	}

	return 0;
}
