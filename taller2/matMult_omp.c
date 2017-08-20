#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matGen.h"

int main(int argc, char const *argv[])
{

	if(argc == 5){
		genMat(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),atoi(argv[4]));
	}

	FILE *f1, *f2, *f3;

	//variables para almacenar las dimensiones de las matrices
	int m1Col, m1Row, m2Col, m2Row;


	f1 = fopen("matg1.txt","r");
	f2 = fopen("matg2.txt","r");
	f3 = fopen("answer.txt","w");
	
	//lectura de dimensiones
	fscanf(f1, "%d", &m1Row); fscanf(f1, "%d", &m1Col);
	fscanf(f2, "%d", &m2Row); fscanf(f2, "%d", &m2Col);

	//condicion de multiplicidad de las matrices
	if (m1Col == m2Row)
	{
		/*asignacion de memoria para los vectores principales
		de cada matriz*/
		int	**mat1;
		mat1 = (int **)malloc(m1Row * sizeof(int *));

		int	**mat2;
		mat2 = (int **)malloc(m2Row * sizeof(int *));


		/*asignacion de memoria para los vectores que conformaran
		las columnas y almacenamiento de valores desde el archivo
		de texto*/
		for (int i = 0; i < m1Row; i++)
		{
			mat1[i] = (int *)malloc(m1Col * sizeof(int));
			for (int j = 0; j < m1Col; j++)
			{
				fscanf(f1, "%d", &mat1[i][j]);
				getc(f1);//saltar las comas (,)
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

		/*dimensiones de la matriz resultado*/
		fprintf(f3, "%d\n", m1Row);
		fprintf(f3, "%d\n", m2Col);

		int a,b;/*variables para almacenar valores a multiplicar*/

		/*ciclo para relizar la multiplicacion de matrices*/

		////////////////////////////////////////////////////////

		printf("\nMat 1 : %d x %d\n", m1Row, m1Col);
		printf("Mat 2 : %d x %d\n", m2Row, m2Col);

		/*INICIO MULTIPLICACION NORMAL*/
		clock_t start1 = clock();
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

	  	/*devuelve una posicion al puntero del archivo resultado,
	  	para no mostar la coma al final de la linea*/
	  	fseek(f3, -1, SEEK_END);

	  	fprintf(f3, "\n");
	 	}
	 	clock_t end1 = clock();
		float seconds1 = (float)(end1 - start1) / CLOCKS_PER_SEC;
		printf("Normal time: %f\n",seconds1);
		/*FIN MULTIPLICACION NORMAL*/

		////////////////////////////////////////////////////////

		/*INICIO MULTIPLICACIÓN CON HILOS*/

		fclose(f3);
		f3 = fopen("answer.txt","w");
		fprintf(f3, "%d\n", m1Row);
		fprintf(f3, "%d\n", m2Col);

		int	tid,nthreads,chunk,i,j,k;
		//chunk = 10;

		clock_t start2 = clock();
		#pragma omp parallel
		
		tid = omp_get_thread_num();
		nthreads = omp_get_num_threads();
		chunk = m1Row / nthreads;

		#pragma omp for
		for (i = 0; i < m1Row; i++) {
	    for (j = 0; j < m2Col; j++) {
	   		int sum = 0;
	      for (k = 0; k < m1Col; k++) {
	        a = mat1[i][k];
	        b = mat2[k][j];
	        sum += a * b;
	      }
	      fprintf(f3, "%d,", sum);
	  	}
	  	/*devuelve una posicion al puntero del archivo resultado,
	  	para no mostar la coma al final de la linea*/
	  	fseek(f3, -1, SEEK_END);

	  	fprintf(f3, "\n");
	 	}

	 	clock_t end2 = clock();
		float seconds2 = (float)(end2 - start2) / CLOCKS_PER_SEC;
		printf("Threading time: %f\n\n",seconds2);
		/*FIN MULTIPLICACION CON HILOS*/

		///////////////////////////////////////////////////////

		fclose(f1); fclose(f2); fclose(f3);

		/*liberacion de memoria*/
		free(mat1); free(mat2);

	}else{
		printf("The arrays can not be multiplied\n");
	}

	return 0;
}
