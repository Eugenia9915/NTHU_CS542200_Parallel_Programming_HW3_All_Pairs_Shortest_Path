#include <sched.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <omp.h>

int main(int argc, char**argv){
    assert(argc==3);
    
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);

    int vertices, edges;
    FILE *ptr = fopen(argv[1],"rb");
    fread(&vertices,sizeof(int),1,ptr);
    fread(&edges,sizeof(int),1,ptr);

    int **board = (int**)malloc(vertices*(sizeof(int*)));
    for(int i = 0; i < vertices; i++)
        board[i]=(int*)malloc(vertices*sizeof(int));
    

    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            if (i != j) board[i][j] = 1073741823; //2^30 - 1
            else board[i][j] = 0;
        }
    }

    int cache[12];
    int cnt = 0;
    while(cnt + 4 < edges){
        fread(cache,sizeof(int),12,ptr);
        board[cache[0]][cache[1]]=cache[2];
        board[cache[3]][cache[4]]=cache[5];
        board[cache[6]][cache[7]]=cache[8];
        board[cache[9]][cache[10]]=cache[11];
        cnt+=4;
    }
    while(cnt < edges){
        fread(cache,sizeof(int),3,ptr);
        board[cache[0]][cache[1]]=cache[2];
        cnt++;
    }
    fclose(ptr);

    //Start Blocked Floyd-Warshall algorithm 

    for(int k = 0;k < vertices;k++){
        #pragma omp parallel for num_threads(ncpus)
            for(int i = 0; i < vertices; i++){
                for(int j = 0; j < vertices; j++){
                    if(board[i][j] > (board[i][k] + board[k][j]))
                        board[i][j] = board[i][k] + board[k][j];
                }
            }
    }
    
    //write
    ptr = fopen(argv[2],"wb");
    for(int i = 0; i < vertices; i++)
        fwrite(&board[i][0], sizeof(int),vertices,ptr);
    fclose(ptr);
    free(board);
}