from openpyxl import load_workbook

path="D:\\PyPath\\other\\excel\\"
# 默认可读写，若有需要可以指定write_only和read_only为True
wb = load_workbook(path+'flush.xlsx')

# 根据sheet名字获得sheet
sheet = wb.get_sheet_by_name('Sheet1')

# 因为按行，所以返回A1, B1, C1这样的顺序
for row in sheet.rows:
    a=""
    for cell in row:
        a=a+str(cell.value)+"    "
    print(a) 
# 修改       
sheet['A1']="名称"

# 保存文件
wb.save(path+'flush_r.xlsx')

# https://www.jianshu.com/p/892023680381