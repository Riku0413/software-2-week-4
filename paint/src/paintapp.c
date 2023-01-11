#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h> // for error catch
#include "object.h"

int main(int argc, char **argv)
{
    //for history recording
    
    const int bufsize = 1000;
    
    // [*]
    History his = (History){ .begin = NULL, .bufsize = bufsize};
    
    int width;
    int height;
    if (argc != 3){
	fprintf(stderr,"usage: %s <width> <height>\n",argv[0]);
	return EXIT_FAILURE;
    } else{
	char *e;
	long w = strtol(argv[1],&e,10);
	if (*e != '\0'){
	    fprintf(stderr, "%s: irregular character found %s\n", argv[1],e);
	    return EXIT_FAILURE;
	}
	long h = strtol(argv[2],&e,10);
	if (*e != '\0'){
	    fprintf(stderr, "%s: irregular character found %s\n", argv[2],e);
	    return EXIT_FAILURE;
	}
	width = (int)w;
	height = (int)h;    
    }
    char pen = '*';
    

    char buf[bufsize];

    Canvas *c = init_canvas(width,height, pen);
    
    printf("\n"); // required especially for windows env

    while(1){
	// [*]
	// hsize 縺ｯ縺ｲ縺ｨ縺ｾ縺壹↑縺�
	// 菴懊ｋ蝣ｴ蜷医�繝ｪ繧ｹ繝磯聞繧定ｪｿ縺ｹ繧矩未謨ｰ繧剃ｽ懊▲縺ｦ縺翫￥
	print_canvas(c);
	printf("* > ");
	if(fgets(buf, bufsize, stdin) == NULL) break;
	
	const Result r = interpret_command(buf, &his, c);

	if (r == EXIT) break;

	// 霑斐▲縺ｦ縺阪◆邨先棡縺ｫ蠢懊§縺ｦ繧ｳ繝槭Φ繝臥ｵ先棡繧定｡ｨ遉ｺ
	clear_command();
	printf("%s\n",strresult(r));
	// LINE縺ｮ蝣ｴ蜷医�History讒矩�菴薙↓蜈･繧後ｋ
	if (r == LINE) {
	    // [*]
	    push_command(&his,buf);
	}
	
	rewind_screen(2); // command results
	clear_command(); // command itself
	rewind_screen(height+2); // rewind the screen to command input
	
    }
    
    clear_screen();
    free_canvas(c);
    
    return 0;
}