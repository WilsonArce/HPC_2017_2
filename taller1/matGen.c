#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char const *argv[])
{
	FILE *f1;
	f1 = fopen("matGen.txt","w");

	int col = atoi(argv[1]);
	int row = atoi(argv[2]);

	time_t t;

	srand((unsigned) time(&t));

	fprintf(f1, "%d\n", col);
	fprintf(f1, "%d\n", row);

	for (int i = 0; i < col; i++) {
  	for (int j = 0; j < row; j++) {
    	fprintf(f1, "%d,", rand() % 100);
  	}
  	fseek(f1, -1, SEEK_END);
  	fprintf(f1, "\n");
	}
	return 0;
}