// Structure for canvas
typedef struct
{
    int width;
    int height;
    char **canvas;
    char pen;
} Canvas;

// Command 讒矩�菴薙→ History讒矩�菴�
// [*]
typedef struct command Command;
struct command{
    char *str;
    size_t bufsize;
    Command *next;
};

typedef struct
{
    Command *begin;
    size_t bufsize; // [*] : 縺薙�譁ｹ縺悟柑邇�噪縺ｧ縺吶�縲ゆｸ驛ｨ縺ｮ譁ｹ縺九ｉ謖�遭縺ゅｊ縺ｾ縺励◆縲�
} History;



// functions for Canvas type
Canvas *init_canvas(int width, int height, char pen);
void reset_canvas(Canvas *c);
void print_canvas(Canvas *c);
void free_canvas(Canvas *c);

// display functions
void rewind_screen(unsigned int line); 
void clear_command(void);
void clear_screen(void);

// enum for interpret_command results
typedef enum res{ EXIT, LINE, UNDO, SAVE, UNKNOWN, ERRNONINT, ERRLACKARGS, NOCOMMAND} Result;
// Result 蝙九↓蠢懊§縺ｦ蜃ｺ蜉帙☆繧九Γ繝�そ繝ｼ繧ｸ繧定ｿ斐☆
char *strresult(Result res);

int max(const int a, const int b);
void draw_line(Canvas *c, const int x0, const int y0, const int x1, const int y1);
Result interpret_command(const char *command, History *his, Canvas *c);
void save_history(const char *filename, History *his);

// [*] list.c 縺ｮpush_back縺ｨ蜷後§
Command *push_command(History *his, const char *str);