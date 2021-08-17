from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, filedialog
import mysql.connector
import csv
import os

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

thongbao =''
tilewin = ''
tileloss = ''
timechay = ''
tile = ''
mydata = []

def update(rows):
	global mydata
	mydata = rows

	trv.delete(*trv.get_children())
	for i in rows:
		trv.insert('', 'end', value=i)


def search():
	q2 = q.get()
	query = "SELECT * FROM NCKH WHERE OPRESULT LIKE '%" + q2 +"%' OR OPRESULT LIKE '%" + q2+"%'"
	cursor.execute(query)
	rows = cursor.fetchall()
	update(rows)


def clear():
	query = "SELECT * FROM NCKH"
	cursor.execute(query)
	rows = cursor.fetchall()
	update(rows)

def getrow(event):
	rowid = trv.identify_row(event.y)
	item = trv.item(trv.focus())
	t1.set(item['values'][0])
	t2.set(item['values'][1])
	t3.set(item['values'][2])
	t4.set(item['values'][3])
	t8.set(item['values'][4])
	t9.set(item['values'][5])
	t10.set(item['values'][6])
	t11.set(item['values'][7])
	t12.set(item['values'][8])
	t13.set(item['values'][9])
	t14.set(item['values'][10])
	t15.set(item['values'][11])
	t16.set(item['values'][12])
	t17.set(item['values'][13])
	t18.set(item['values'][14])
	t19.set(item['values'][15])
	t20.set(item['values'][16])
	t21.set(item['values'][17])
	t22.set(item['values'][18])
	

def update_customer():
	g1 = t1.get()
	g2 = t2.get()
	g3 = t3.get()
	g4 = t4.get()
	g5 = t8.get()
	g6 = t9.get()
	g7 = t10.get()
	g8 = t11.get()
	g9 = t12.get()
	g10 = t13.get()
	g11 = t14.get()
	g12 = t15.get()
	g13 = t16.get()
	g14 = t17.get()
	g15 = t18.get()
	g16 = t19.get()
	g17 = t20.get()
	g18 = t21.get()
	g19 = t22.get()
	
	if messagebox.askyesno("Cap Nhat?", "Ban co chac khong?"):
		query = "UPDATE NCKH SET SSUP = %s,SUPGROUP =%s,REGION=%s,RTM=%s,EDISS=%s,OPRESULT=%s,SSCC=%s,TDITC=%s,TDITQ=%s,OAUSD=%s,CZBR=%s,CZBEC=%s,RFCPTY=%s,COMTYPE=%s,RDITTD=%s,RDVTTD=%s,RDQTTD=%s,DSC = %s WHERE OPNUMBER = %s"
		cursor.execute(query,(g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12,g13,g14,g15,g16,g17,g18,g19,g1))
		mydb.commit()
		clear()
	else:
		return True


def add_new():
	gt1 = t1.get()
	gt2 = t2.get()
	gt3 = t3.get()
	gt4 = t4.get()
	gt5 = t8.get()
	gt6 = t9.get()
	gt7 = t10.get()
	gt8 = t11.get()
	gt9 = t12.get()
	gt10 = t13.get()
	gt11 = t14.get()
	gt12 = t15.get()
	gt13 = t16.get()
	gt14 = t17.get()
	gt15 = t18.get()
	gt16 = t19.get()
	gt17 = t20.get()
	gt18 = t21.get()
	gt19 = t22.get()
	query = "INSERT INTO NCKH VALUES(%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s)"
	cursor.execute(query, (gt1,gt2,gt3,gt4,gt5,gt6,gt7,gt8,gt9,gt10,gt11,gt12,gt13,gt14,gt15,gt16,gt17,gt18,gt19))
	mydb.commit()
	clear()
def delete_customer():
	customer_id = t1.get()
	if messagebox.askyesno("Xoa", "Ban co chac khong?"):
		query = "DELETE FROM NCKH WHERE OPNUMBER = " + customer_id
		cursor.execute(query)
		mydb.commit()
		clear()
	else:
		return True

def export():
	if len(mydata) < 1:
		messagebox.showerror("No data", "No data availabel to export")
		return False

	fln = filedialog.asksaveasfilename(initialdir=os.getcwd(), title="Save CSV", filetypes=(("CSV File", "*.csv"), ("All Files", "*.*")))
	with open(fln, mode='w') as myfile:
		exp_writer = csv.writer(myfile, delimiter =',')
		for i in mydata:
			exp_writer.writerow(i)
	messagebox.showinfo("Data Exported", "csv : " + os.path.basename(fln)+" thanh cong.")

def importcsv():
	mydata.clear()
	fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Open CSV", filetypes = (("CSV File", "*.csv"), ("All Files", "*.*")))
	with open(fln) as myfile:
		csvread = csv.reader(myfile, delimiter =",")
		for i in csvread:
			mydata.append(i)
	update(mydata)
	return fln

def savedb():
	if messagebox.askyesno("Save data", "Ban co muon luu data vao database:"):
		for i in mydata:
			gt1 = i[0]
			gt2 = i[1]
			gt3 = i[2]
			gt4 = i[3]
			gt5 = i[4]
			gt6 = i[5]
			gt7 = i[6]
			gt8 = i[7]
			gt9 = i[8]
			gt10 = i[9]
			gt11 = i[10]
			gt12 = i[11]
			gt13 = i[12]
			gt14 = i[13]
			gt15 = i[14]
			gt16 = i[15]
			gt17 = i[16]
			gt18 = i[17]
			gt19 = i[18]
			query = "INSERT INTO NCKH VALUES(%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s)"
			cursor.execute(query, (gt1,gt2,gt3,gt4,gt5,gt6,gt7,gt8,gt9,gt10,gt11,gt12,gt13,gt14,gt15,gt16,gt17,gt18,gt19))
		mydb.commit()
		clear()
		messagebox.showinfo("Data save","Luu thanh cong")
	else:
		return False

def dudoan():
	link = importcsv()
	with open(link) as mylink:
		sales_data = pd.read_csv(mylink)
		sales_data_original = sales_data
		number_of_nulls_by_column = sales_data.isnull().sum().sort_values(ascending=False)
		percent_of_nulls = sales_data.isnull().sum()/sales_data.isnull().count()*100
		percent_rounded = (round(percent_of_nulls, 1)).sort_values(ascending=False)
		missing_data = pd.concat([number_of_nulls_by_column, percent_rounded], axis=1, keys=['Total', '%'])
		le = LabelEncoder()
		sales_data['Supplies Subgroup'] = le.fit_transform(sales_data['Supplies Subgroup'])
		sales_data['Region'] = le.fit_transform(sales_data['Region'])
		sales_data['Route To Market'] = le.fit_transform(sales_data['Route To Market'])
		sales_data['Opportunity Result'] = le.fit_transform(sales_data['Opportunity Result'])
		sales_data['Competitor Type'] = le.fit_transform(sales_data['Competitor Type'])
		sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])

		numeric_features = sales_data.select_dtypes(include=[np.number])

		corr = numeric_features.corr()

		#traning
		cols = [col for col in sales_data.columns if col not in ['Opportunity Number','Opportunity Result']]

		data = sales_data[cols]
		target = sales_data['Opportunity Result']
		#cat data de train va test 
		data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)
		#random forest
		#as
		start = time.time()
		random_forest = RandomForestClassifier()
		pred = random_forest.fit(data_train, target_train).predict(data_test)

		#in điểm chính xác của mô hìnhogistic Random Forest
		from sklearn.naive_bayes import GaussianNB
		from sklearn.metrics import accuracy_score
		acc_random_forest = round(accuracy_score(target_test, pred, normalize = True)*100, 2)
		end = time.time()

		tile = "Độ chính xác Random Forest :", acc_random_forest, "%"
		timechay = "Thời gian chạy:", end - start, "giây"
		t6.set(tile)
		t7.set(timechay)
		#check diem
		from sklearn.model_selection import cross_val_score
		rf = RandomForestClassifier()
		scores = cross_val_score(rf, data_train, target_train, cv=10, scoring = "accuracy")
		results = pd.DataFrame({"Scores:":scores, 
                		        "Mean:":scores.mean(), 
                        		"Standard Deviation:":scores.std()})
		#tiep tuc training
		train_df  = data_train.drop("Client Size By Employee Count", axis=1)
		test_df  = data_test.drop("Client Size By Employee Count", axis=1)

		train_df  = train_df.drop("Competitor Type", axis=1)
		test_df  = test_df.drop("Competitor Type", axis=1)
		#Random forest acc_random_forest
		random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
		random_forest.fit(data_train, target_train)
		Y_prediction = random_forest.predict(data_train)
		random_forest.score(data_train, target_train)
		#Tiep tuc training
		random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

		random_forest.fit(data_train, target_train)
		Y_prediction = random_forest.predict(data_test)
		random_forest.score(data_train, target_train)
		#matrix
		from sklearn.model_selection import cross_val_predict
		from sklearn.metrics import confusion_matrix
		predictions = cross_val_predict(random_forest, data_train, target_train, cv=3)
		confusion_matrix(target_train, predictions)
		#tra ve ket qua
		won = sales_data.loc[sales_data['Opportunity Result'] == 1].shape[0]/(sales_data.loc[sales_data['Opportunity Result'] == 1].shape[0] + sales_data.loc[sales_data['Opportunity Result'] == 0].shape[0]) * 100
		loss = sales_data.loc[sales_data['Opportunity Result'] == 0].shape[0]/(sales_data.loc[sales_data['Opportunity Result'] == 1].shape[0] + sales_data.loc[sales_data['Opportunity Result'] == 0].shape[0]) * 100
		#ket luan
		new_data_instance = ["Exterior Accessories", "Car Accessories", "Northwest",
                     		"Fields Sales", 76, 13, 104, 101, 0, 5, 5, 0, "Unknown", 0.69636, 0.113985, 0.154215, 1]

		new_data_instance = le.fit_transform(new_data_instance)

		inference = random_forest.predict([new_data_instance])
		if inference[0] == 0:
			thongbao = "Nếu không thay đổi phương pháp kinh doanh, công ty có nguy cơ lỗ đến: ",  round(((round(loss, 2))-round(won, 2)),2), "%"
			t23.set(thongbao)
			tileloss = "Dự đoán: tỉ lệ lỗ: ",(round(loss, 2)),"%", "và tỉ lệ lãi: ",(round(won, 2)),"%"
			t5.set(tileloss)
			warning = "****Lưu ý: Tất cả số liệu do máy dự đoán có thể có sai số ****"
			t24.set(warning)
		elif inference[0] == 1:
			tilewin = "Dự đoán: tỉ lệ lãi: ",(round(won, 2)),"%", "và tỉ lệ lỗ:" , (round(loss, 2)),"%"
			t5.set(tilewin)
			thongbao = "Tiếp tục kinh doanh như vậy, tỉ lệ lãi đang tăng: "
			t23.set(thongbao)
			warning = "****Lưu ý: Tất cả số liệu do máy dự đoán có thể có sai số ****"
			t24.set(warning)
		else:
    			print(inference)

mydb = mysql.connector.connect(host="localhost", user="root", passwd="", database="NCKH")
cursor = mydb.cursor()

root = Tk()
q = StringVar()
t1 = StringVar()
t2 = StringVar()
t3 = StringVar()
t4 = StringVar()

t6 = StringVar()
t7 = StringVar()

t8 = StringVar()
t9 = StringVar()
t10 = StringVar()
t11 = StringVar()
t12 = StringVar()
t13 = StringVar()
t14 = StringVar()
t15 = StringVar()
t16 = StringVar()
t17 = StringVar()
t18 = StringVar()
t19 = StringVar()
t20 = StringVar()
t21 = StringVar()
t22 = StringVar()
t23 = StringVar()
t5 = StringVar()
t24 = StringVar()


wrapper1 = LabelFrame(root, text ="Danh sách dữ liệu")
wrapper2 = LabelFrame(root, text ="Tim Kiem")
wrapper3 = LabelFrame(root, text ="Chỉnh sửa dữ liệu")

wrapper1.pack(fill="both", expand="yes", padx=20, pady=20)
wrapper2.pack(fill="both", expand="yes", padx=20, pady=10)
wrapper3.pack(fill="both", expand="yes", padx=20, pady=10)

trv = ttk.Treeview(wrapper1, columns=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19), show="headings", height="6")
trv.pack()
verscrlbar = ttk.Scrollbar(root, 
                           orient ="vertical", 
                           command = trv.yview)
verscrlbar.pack(side ='right', fill ='x')

trv.configure(xscrollcommand = verscrlbar.set)

trv.heading(1, text="Opportunity Number")
trv.heading(2, text="Supplies Subgroup")
trv.heading(3, text="Supplies Group")
trv.heading(4, text="Region")
trv.heading(5, text="Route To Market")
trv.heading(6, text="Elapsed Days In Sales Stage")
trv.heading(7, text="Opportunity Result")
trv.heading(8, text="Sales Stage Change Count")
trv.heading(9, text="Total Days Identified Through Closing")
trv.heading(10, text="Total Days Identified Through Qualified")
trv.heading(11, text="Opportunity Amount USD")
trv.heading(12, text="Client Size By Revenue")
trv.heading(13, text="Client Size By Employee Count")
trv.heading(14, text="Revenue From Client Past Two Years")
trv.heading(15, text="Competitor Type")
trv.heading(16, text="Ratio Days Identified To Total Days")
trv.heading(17, text="Ratio Days Validated To Total Days")
trv.heading(18, text="Ratio Days Qualified To Total Days")
trv.heading(19, text="Deal Size Category")

trv.bind('<Double 1>', getrow)

expbtn = Button(wrapper1, text= "Xuất dữ liệu CSV", command=export)
expbtn.pack(side=tk.LEFT, padx=10, pady=10)

impbtn = Button(wrapper1, text= "Chèn dữ liệu CSV", command=importcsv)
impbtn.pack(side=tk.LEFT, padx=10, pady=10)

savebtn = Button(wrapper1, text= "Lưu dữ liệu", command=savedb)
savebtn.pack(side=tk.LEFT, padx=10, pady=10)

dudoanbtn = Button(wrapper1, text= "Dự đoán", command=dudoan)
dudoanbtn.pack(side = tk.LEFT, padx=10, pady=10)

exbtn = Button(wrapper1, text= "Thoát", command=lambda: exit())
exbtn.pack(side=tk.LEFT, padx=10, pady=10)


query = "SELECT * from NCKH"
cursor.execute(query)
rows = cursor.fetchall()
update(rows)



#btn du doan 
lbl6 = Label(wrapper1, textvariable=t6)
lbl6.pack(side=tk.BOTTOM, padx=20, pady=10)
lbl7 = Label(wrapper1, textvariable=t7)
lbl7.pack(side=tk.BOTTOM, padx=20, pady=10)
lbl99 = Label(wrapper1, textvariable=t23)
lbl99.pack(side=tk.BOTTOM, padx=20, pady=10)
lbl5 = Label(wrapper1, textvariable=t5)
lbl5.pack(side=tk.BOTTOM, padx=20, pady=10)
lbl999 = Label(wrapper1, textvariable=t24)
lbl999.pack(side=tk.BOTTOM, padx=20, pady=10)
#ent5 = Entry(wrapper1, textvariable=t5)
#ent5.pack(side=tk.LEFT,  padx =0, pady=50)



#search section 
lbl = Label(wrapper2, text="Tìm kiếm")
lbl.pack(side=tk.LEFT, padx=10)
ent= Entry(wrapper2, textvariable=q)
ent.pack(side=tk.LEFT, padx=6)
btn = Button(wrapper2, text="Tìm kiếm", command=search)
btn.pack(side=tk.LEFT, padx=6)
cbtn = Button(wrapper2, text="Hủy tìm kiếm", command=clear)
cbtn.pack(side=tk.LEFT, padx=6)

#user data section
lbl1 = Label(wrapper3, text="OP Number")
lbl1.grid(row=0, column=0, padx=5, pady=3)
ent1 = Entry(wrapper3, textvariable=t1)
ent1.grid(row=0, column=1, padx =5, pady=3)

lbl2 = Label(wrapper3, text="Supplies Subgroup")
lbl2.grid(row=1, column=0, padx=5, pady=3)
ent2 = Entry(wrapper3, textvariable=t2)
ent2.grid(row=1, column=1, padx =5, pady=3)

lbl3 = Label(wrapper3, text="Supplies Group")
lbl3.grid(row=2, column=0, padx=5, pady=3)
ent3 = Entry(wrapper3, textvariable=t3)
ent3.grid(row=2, column=1, padx =5, pady=3)

lbl4 = Label(wrapper3, text="Region")
lbl4.grid(row=3, column=0, padx=5, pady=3)
ent4 = Entry(wrapper3, textvariable=t4)
ent4.grid(row=3, column=1, padx =5, pady=3)

lbl5 = Label(wrapper3, text="Route To Market")
lbl5.grid(row=4, column=0, padx=5, pady=3)
ent5 = Entry(wrapper3, textvariable=t8)
ent5.grid(row=4, column=1, padx =5, pady=3)

lbl6 = Label(wrapper3, text="Elapsed Day In slase Stage")
lbl6.grid(row=5, column=0, padx=5, pady=3)
ent6 = Entry(wrapper3, textvariable=t9)
ent6.grid(row=5, column=1, padx =5, pady=3)

lbl7 = Label(wrapper3, text="Opportunity Result")
lbl7.grid(row=0, column=2, padx=5, pady=3)
ent7 = Entry(wrapper3, textvariable=t10)
ent7.grid(row=0, column=3, padx =5, pady=3)

lbl8 = Label(wrapper3, text="Sales Stage Change Count")
lbl8.grid(row=1, column=2, padx=5, pady=3)
ent8 = Entry(wrapper3, textvariable=t11)
ent8.grid(row=1, column=3, padx =5, pady=3)

lbl9 = Label(wrapper3, text="Total-D-I-T-closing")
lbl9.grid(row=2, column=2, padx=5, pady=3)
ent9 = Entry(wrapper3, textvariable=t12)
ent9.grid(row=2, column=3, padx =5, pady=3)

lbl10 = Label(wrapper3, text="Total-D-I-T-Qualified")
lbl10.grid(row=3, column=2, padx=5, pady=3)
ent10 = Entry(wrapper3, textvariable=t13)
ent10.grid(row=3, column=3, padx =5, pady=3)

lbl11 = Label(wrapper3, text="Opportunity Amount USD")
lbl11.grid(row=4, column=2, padx=5, pady=3)
ent11 = Entry(wrapper3, textvariable=t14)
ent11.grid(row=4, column=3, padx =5, pady=3)

lbl12 = Label(wrapper3, text="Client Size By Revenue")
lbl12.grid(row=5, column=2, padx=5, pady=3)
ent12 = Entry(wrapper3, textvariable=t15)
ent12.grid(row=5, column=3, padx =5, pady=3)

lbl13 = Label(wrapper3, text="Client-S-B-Emp-count")
lbl13.grid(row=0, column=4, padx=5, pady=3)
ent13 = Entry(wrapper3, textvariable=t16)
ent13.grid(row=0, column=5, padx =5, pady=3)

lbl14 = Label(wrapper3, text="Revenue From Client Past Two Years")
lbl14.grid(row=1, column=4, padx=5, pady=3)
ent14 = Entry(wrapper3, textvariable=t17)
ent14.grid(row=1, column=5, padx =5, pady=3)

lbl15 = Label(wrapper3, text="Competitor Type")
lbl15.grid(row=2, column=4, padx=5, pady=3)
ent15 = Entry(wrapper3, textvariable=t18)
ent15.grid(row=2, column=5, padx =5, pady=3)

lbl16 = Label(wrapper3, text="Ration-D-Indentify-to-total-day")
lbl16.grid(row=3, column=4, padx=5, pady=3)
ent16 = Entry(wrapper3, textvariable=t19)
ent16.grid(row=3, column=5, padx =5, pady=3)

lbl17 = Label(wrapper3, text="Ration-D-Validate-to-total-day")
lbl17.grid(row=4, column=4, padx=5, pady=3)
ent17 = Entry(wrapper3, textvariable=t20)
ent17.grid(row=4, column=5, padx =5, pady=3)

lbl18 = Label(wrapper3, text="Ratio-D-Qualified-to-total-day")
lbl18.grid(row=5, column=4, padx=5, pady=3)
ent18 = Entry(wrapper3, textvariable=t21)
ent18.grid(row=5, column=5, padx =5, pady=3)

lbl19 = Label(wrapper3, text="Deal Size Category")
lbl19.grid(row=6, column=4, padx=5, pady=3)
ent19 = Entry(wrapper3, textvariable=t22)
ent19.grid(row=6, column=5, padx =5, pady=3)




up_btn = Button(wrapper2, text="Cập nhật", command=update_customer)
add_btn = Button(wrapper2, text="Thêm mới", command=add_new)
delete_btn = Button(wrapper2, text="Xóa bỏ", command=delete_customer)

add_btn.pack(side=tk.RIGHT, padx=5, pady=3)
up_btn.pack(side=tk.RIGHT, padx=5, pady=3)
delete_btn.pack(side=tk.RIGHT,padx=5, pady=3)



root.title("Dự đoán tỉ lệ lãi lỗ")
root.geometry("1240x700")
root.mainloop()

