#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void genMat(int col1, int row1, int col2, int row2){

	time_t t;
	srand((unsigned) time(NULL));

	FILE *f1, *f2;
	f1 = fopen("matg1.txt","w");
	f2 = fopen("matg2.txt","w");

	printf("\nGenerating arrays...");
	fprintf(f1, "%d\n", col1);
	fprintf(f1, "%d\n", row1);

	for (int i = 0; i < col1; i++) {
  	for (int j = 0; j < row1; j++) {
    	fprintf(f1, "%d,", rand() % 100);
  	}
  	fseek(f1, -1, SEEK_END);
  	fprintf(f1, "\n");
	}

	fprintf(f2, "%d\n", col2);
	fprintf(f2, "%d\n", row2);

	for (int i = 0; i < col2; i++) {
  	for (int j = 0; j < row2; j++) {
    	fprintf(f2, "%d,", rand() % 100);
  	}
  	fseek(f2, -1, SEEK_END);
  	fprintf(f2, "\n");
	}


	fclose(f1); fclose(f2);

}
