#include <iostream>
#include <string>
#include<time.h>
using namespace std;

char get_rand_char() {//借助随机库函数随机生成一个字符串
    char str[64] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    //srand((unsigned int)time((time_t *)NULL));  如果这条语句放在这里，则在get_rand_str()中每次产生的char都是一样的
    return str[rand() % 52];
}
/*把随机生成的字符串插入到已经定义好的set<string> s中去*/
char get_rand_char();
string str = "\0";

srand((unsigned int)time((time_t *)NULL));//time seed created

for (int j = 0; j < SIZE; j++) {//SIZE==53
    for (int i = 0; i < length; i++) {//length is the length of the strings
        str.push_back(get_rand_char());
    }
    s.insert(str);
    str = "\0";
}



string T[53] = { "tactagcaatacgcttgcgttcggtggttaagtatgtataatgcgcgggcttgtcgt",
    "tgctatcctgacagttgtcacgctgattggtgtcgttacaatctaacgcatcgccaa",
    "gtactagagaactagtgcattagcttatttttttgttatcatgctaaccacccggcg",
    "aattgtgatgtgtatcgaagtgtgttgcggagtagatgttagaatactaacaaactc",
    "tcgataattaactattgacgaaaagctgaaaaccactagaatgcgcctccgtggtag",
    "aggggcaaggaggatggaaagaggttgccgtataaagaaactagagtccgtttaggt",
    "cagggggtggaggatttaagccatctcctgatgacgcatagtcagcccatcatgaat",
    "tttctacaaaacacttgatactgtatgagcatacagtataattgcttcaacagaaca",
    "cgacttaatatactgcgacaggacgtccgttctgtgtaaatcgcaatgaaatggttt",
    "ttttaaatttcctcttgtcaggccggaataactccctataatgcgccaccactgaca",
    "gcaaaaataaatgcttgactctgtagcgggaaggcgtattatgcacaccccgcgccg",
    "cctgaaattcagggttgactctgaaagaggaaagcgtaatatacgccacctcgcgac",
    "gatcaaaaaaatacttgtgcaaaaaattgggatccctataatgcgcctccgttgaga",
    "ctgcaatttttctattgcggcctgcggagaactccctataatgcgcctccatcgaca",
    "tttatatttttcgcttgtcaggccggaataactccctataatgcgccaccactgaca",
    "aagcaaagaaatgcttgactctgtagcgggaaggcgtattatgcacaccgccgcgcc",
    "atgcatttttccgcttgtcttcctgagccgactccctataatgcgcctccatcgaca",
    "aaacaatttcagaatagacaaaaactctgagtgtaataatgtagcctcgtgtcttgc",
    "tctcaacgtaacactttacagcggcgcgtcatttgatatgatgcgccccgcttcccg",
    "gcaaataatcaatgtggacttttctgccgtgattatagacacttttgttacgcgttt",
    "gacaccatcgaatggcgcaaaacctttcgcggtatggcatgatagcgcccggaagag",
    "aaaaacgtcatcgcttgcattagaaaggtttctggccgaccttataaccattaatta",
    "tctgaaatgagctgttgacaattaatcatcgaactagttaactagtacgcaagttca",
    "accggaagaaaaccgtgacattttaacacgtttgttacaaggtaaaggcgacgccgc",
    "aaattaaaattttattgacttaggtcactaaatactttaaccaatataggcatagcg",
    "ttgtcataatcgacttgtaaaccaaattgaaaagatttaggtttacaagtctacacc",
    "catcctcgcaccagtcgacgacggtttacgctttacgtatagtggcgacaatttttt",
    "tccagtataatttgttggcataattaagtacgacgagtaaaattacatacctgcccg",
    "acagttatccactattcctgtggataaccatgtgtattagagttagaaaacacgagg",
    "tgtgcagtttatggttccaaaatcgccttttgctgtatatactcacagcataactgt",
    "ctgttgttcagtttttgagttgtgtataacccctcattctgatcccagcttatacgg",
    "attacaaaaagtgctttctgaactgaacaaaaaagagtaaagttagtcgcgtagggt",
    "atgcgcaacgcggggtgacaagggcgcgcaaaccctctatactgcgcgccgaagctg",
    "taaaaaactaacagttgtcagcctgtcccgcttataagatcatacgccgttatacgt",
    "atgcaattttttagttgcatgaactcgcatgtctccatagaatgcgcgctacttgat",
    "ccttgaaaaagaggttgacgctgcaaggctctatacgcataatgcgccccgcaacgc",
    "tcgttgtatatttcttgacaccttttcggcatcgccctaaaattcggcgtcctcata",
    "ccgtttattttttctacccatatccttgaagcggtgttataatgccgcgccctcgat",
    "ttcgcatatttttcttgcaaagttgggttgagctggctagattagccagccaatctt",
    "tgtaaactaatgcctttacgtgggcggtgattttgtctacaatcttacccccacgta",
    "gatcgcacgatctgtatacttatttgagtaaattaacccacgatcccagccattctt",
    "aacgcatacggtattttaccttcccagtcaagaaaacttatcttattcccacttttc",
    "ttagcggatcctacctgacgctttttatcgcaactctctactgtttctccatacccg",
    "gccttctccaaaacgtgttttttgttgttaattcggtgtagacttgtaaacctaaat",
    "cagaaacgttttattcgaacatcgatctcgtcttgtgttagaattctaacatacggt",
    "cactaatttattccatgtcacacttttcgcatctttgttatgctatggttatttcat",
    "atataaaaaagttcttgctttctaacgtgaaagtggtttaggttaaaagacatcagt",
    "caaggtagaatgctttgccttgtcggcctgattaatggcacgatagtcgcatcggat",
    "ggccaaaaaatatcttgtactatttacaaaacctatggtaactctttaggcattcct",
    "taggcaccccaggctttacactttatgcttccggctcgtatgttgtgtggaattgtg",
    "ccatcaaaaaaatattctcaacataaaaaactttgtgtaatacttgtaacgctacat",
    "tggggacgtcgttactgatccgcacgtttatgatatgctatcgtactctttagcgag",
    "tcagaaatattatggtgatgaactgtttttttatccagtataatttgttggcataat",
};

for (int i = 0; i < SIZE; i++) {
    s.insert(T[i]);

cout << "请您输入阈值（0-1）:" << endl;
while (1) {
    cin >> threshold;
    if (threshold > 0 && threshold < 1)
        break;
    else
        cout << "输入错误！请您输入阈值（0-1）:" << endl;
}
double number = SIZE*threshold;//达到阈值的字符串临界值数量，包含该子串的字符串数量小于这个值则不输出，大于等于就输出
cout << "子字符串至少需要出现" << number << "次才输出。" << endl;



/*--------------------------找出字符串中的所有子串---------------------------*/
set<string>::iterator it = s.begin();
set<string> s_sub;//满足符合条件的子串集合
string sub_str = "\0";//存储字符串中的子串
for (int i = 0; i < SIZE; i++) {//对字符串集合中的SIZE个字符串循环
    for (int j = 0; j < SUBNUM; ) {//SUBNUM为给定长度length的字符串中所包含的所有子串数目，其值可以用数学方法算出来等于(length^2-4)/2
        string::const_iterator p = (*it).begin();
        for (int n = 1; n <= length-1; n++) {
            for (int k = 0; k <= length - n; k++) {
                for (int c = 0; c < n; c++) {
                    sub_str.push_back(p[k + c]);
                }
                j++;
                int count = 0;
                for (set<string>::iterator q = s.begin(); q != s.end(); q++) {
                    if ((*q).find(sub_str) != string::npos) {
                        count++;
                    }
                }
                string temp = "\0";
                temp = sub_str;
                sub_str = "\0";
                if (count >= number) {
                    s_sub.insert(temp);
                }
            }

            p = (*it).begin();
        }
    }
    ++it;
}


int num = 0;
for (set<string>::iterator i = s_sub.begin(); i != s_sub.end(); i++) {
    cout << *i << "\t";
    num++;
}
cout << "共有" << num << "个子串" << endl;
cout << endl << endl;
it = s.begin();
for (it; it != s.end(); it++) {
    cout << *it << '\n';
}

clock_t start, finish;
start = clock();
/*算法执行*/

/*执行结束*/
finish = clock();
cout << "运行时间为：" << (finish - start) / CLOCKS_PER_SEC << "s" << endl;

