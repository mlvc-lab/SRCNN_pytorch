#include <iostream>
using namespace std;

int main() {
	int num = 0;
	float x, y, result;
	
	while (num != 5) {
		system("cls");
		cout << "===== 건방진 계산기 =====" << endl;
		cout << "1. 덧셈" << endl;
		cout << "2. 뺄셈" << endl;
		cout << "3. 곱셈" << endl;
		cout << "4. 나눗셈" << endl;
		cout << "5. 나가기" << endl; 
		cout << "=========================" << endl;
		cout << "번호를 입력해라. : ";
		cin >> num;
               
		if( num >= 1 || num <= 4) {
		    cout << "수를 두 개 입력해라." << endl;
	        cin >> x >> y;
		}
		
		switch (num) {
		case 1:
			cout << "답은 " << x+y << endl;
			system("pause");
			break;
		case 2:
			cout << "답은 " << x-y << endl;
			system("pause");
			break;
		case 3:
			cout << "답은 " << x*y << endl;
			system("pause");
			break;
		case 4:
			cout << "답은 " << x/y << endl;
			system("pause");
			break;
		case 5:
			cout << "이제 가라." << endl;
			break;
		default:
			cout << "이상한걸 입력했구나." << endl;
			system("pause");
			break;
		}
	}
	
}