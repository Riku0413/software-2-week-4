#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "object.h"

Canvas *init_canvas(int width,int height, char pen)
{
    Canvas *new = (Canvas *)malloc(sizeof(Canvas));
    new->width = width;
    new->height = height;
    new->canvas = (char **)malloc(width * sizeof(char *));
    
    char *tmp = (char *)malloc(width*height*sizeof(char));
    memset(tmp, ' ', width*height*sizeof(char));
    for (int i = 0 ; i < width ; i++){
	new->canvas[i] = tmp + i * height;
    }
    
    new->pen = pen;
    return new;
}

void reset_canvas(Canvas *c)
{
    const int width = c->width;
    const int height = c->height;
    memset(c->canvas[0], ' ', width*height*sizeof(char));
}


void print_canvas(Canvas *c)
{
    const int height = c->height;
    const int width = c->width;
    char **canvas = c->canvas;
    
    // 荳翫�螢�
    printf("+");
    for (int x = 0 ; x < width ; x++)
	printf("-");
    printf("+\n");
    
    // 螟門｣√→蜀��
    for (int y = 0 ; y < height ; y++) {
	printf("|");
	for (int x = 0 ; x < width; x++){
	    const char c = canvas[x][y];
	    putchar(c);
	}
	printf("|\n");
    }
    
    // 荳九�螢�
    printf( "+");
    for (int x = 0 ; x < width ; x++)
	printf("-");
    printf("+\n");
    fflush(stdout);
}

void free_canvas(Canvas *c)
{
    free(c->canvas[0]); //  for 2-D array free
    free(c->canvas);
    free(c);
}

void rewind_screen(unsigned int line)
{
    printf("\e[%dA",line);
}

void clear_command(void)
{
    printf("\e[2K");
}

void clear_screen(void)
{
    printf( "\e[2J");
}


int max(const int a, const int b)
{
    return (a > b) ? a : b;
}
void draw_line(Canvas *c, const int x0, const int y0, const int x1, const int y1)
{
    const int width = c->width;
    const int height = c->height;
    char pen = c->pen;
    
    const int n = max(abs(x1 - x0), abs(y1 - y0));
    if ( (x0 >= 0) && (x0 < width) && (y0 >= 0) && (y0 < height))
	c->canvas[x0][y0] = pen;
    for (int i = 1; i <= n; i++) {
	const int x = x0 + i * (x1 - x0) / n;
	const int y = y0 + i * (y1 - y0) / n;
	if ( (x >= 0) && (x< width) && (y >= 0) && (y < height))
	    c->canvas[x][y] = pen;
    }
}

void save_history(const char *filename, History *his)
{
    const char *default_history_file = "history.txt";
    if (filename == NULL)
	filename = default_history_file;
    
    FILE *fp;
    if ((fp = fopen(filename, "w")) == NULL) {
	fprintf(stderr, "error: cannot open %s.\n", filename);
	return;
    }
    // [*] 邱壼ｽ｢繝ｪ繧ｹ繝育沿
    for (Command *p = his->begin ; p != NULL ; p = p->next){
	fprintf(fp, "%s", p->str);
    }
    
    fclose(fp);
}

Result interpret_command(const char *command, History *his, Canvas *c)
{
    char buf[his->bufsize];
    strcpy(buf, command);
    buf[strlen(buf) - 1] = 0; // remove the newline character at the end
    
    const char *s = strtok(buf, " ");
    if (s == NULL){ // 謾ｹ陦後□縺大�蜉帙＆繧後◆蝣ｴ蜷�
	return UNKNOWN;
    }
    // The first token corresponds to command
    if (strcmp(s, "line") == 0) {
	int p[4] = {0}; // p[0]: x0, p[1]: y0, p[2]: x1, p[3]: x1 
	char *b[4];
	for (int i = 0 ; i < 4; i++){
	    b[i] = strtok(NULL, " ");
	    if (b[i] == NULL){
		return ERRLACKARGS;
	    }
	}
	for (int i = 0 ; i < 4 ; i++){
	    char *e;
	    long v = strtol(b[i],&e, 10);
	    if (*e != '\0'){
		return ERRNONINT;
	    }
	    p[i] = (int)v;
	}
	
	draw_line(c,p[0],p[1],p[2],p[3]);
	return LINE;
    }
    
    if (strcmp(s, "save") == 0) {
	s = strtok(NULL, " ");
	save_history(s, his);
	return SAVE;
    }
    
    if (strcmp(s, "undo") == 0) {
	reset_canvas(c);
	//[*] 邱壼ｽ｢繝ｪ繧ｹ繝医�蜈磯�ｭ縺九ｉ繧ｹ繧ｭ繝｣繝ｳ縺励※騾先ｬ｡螳溯｡�
	// pop_back 縺ｮ繧ｹ繧ｭ繝｣繝ｳ荳ｭ縺ｫinterpret_command 繧堤ｵ｡繧√◆諢溘§
	Command *p = his->begin;
	if (p == NULL){
	    return NOCOMMAND;
	}
	else{
	    Command *q = NULL; // 譁ｰ縺溘↑邨らｫｯ繧呈ｱｺ繧√ｋ譎ゅ↓菴ｿ縺�
	    while (p->next != NULL){ // 邨らｫｯ縺ｧ縺ｪ縺�さ繝槭Φ繝峨�螳溯｡後＠縺ｦ濶ｯ縺�
		interpret_command(p->str, his, c);
		q = p;
		p = p->next;
	    }
	    // 1縺､縺励°縺ｪ縺�さ繝槭Φ繝峨�undo縺ｧ縺ｯ繝ｪ繧ｹ繝医�蜈磯�ｭ繧貞､画峩縺吶ｋ
	    if (q == NULL) {
		his->begin = NULL;
	    }
	    else{
		q->next = NULL;
	    }
	    free(p->str);
	    free(p);	
	    return UNDO;
	}  
    }
    
    if (strcmp(s, "quit") == 0) {
	return EXIT;
    }
    return UNKNOWN;
}


// [*] 邱壼ｽ｢繝ｪ繧ｹ繝医�譛ｫ蟆ｾ縺ｫpush 縺吶ｋ
Command *push_command(History *his, const char *str){
    Command *c = (Command*)malloc(sizeof(Command));
    char *s = (char*)malloc(his->bufsize);
    strcpy(s, str);
    
    *c = (Command){ .str = s, .bufsize = his->bufsize, .next = NULL};
    
    Command *p = his->begin;
    
    if ( p == NULL) {
	his->begin = c;
    }
    else{
	while (p->next != NULL){
	    p = p->next;
	}
	p->next = c;
    }
    return c;
}

char *strresult(Result res){
    switch(res) {
    case EXIT:
	break;
    case SAVE:
	return "history saved";
    case LINE:
	return "1 line drawn";
    case UNDO:
	return "undo!";
    case UNKNOWN:
	return "error: unknown command";
    case ERRNONINT:
	return "Non-int value is included";
    case ERRLACKARGS:
	return "Too few arguments";
    case NOCOMMAND:
	return "No command in history";
    }
    return NULL;
}