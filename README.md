# Stock-Index-Forecasting-for-Trading-Web
เว็บแอปพลิเคชั่นสําหรับทํานายแนวโน้มราคาในตลาดตราสารทุน

myproject --> myapp --> 
- preprocessstock.py ดาวน์โหลดและจัดรูปข้อมูล
- predictor.py เรียกใช้งาน model
- batch_manager.py จัดการ batch
- plot.py สร้างกราฟ
- views.py เรียกใช้งานฟังก์ชั่นทั้งหมด และส่งออกไปแสดงยังหน้าเว็บ

--> templates เก็บบันทึกไฟล์ html
--> static เก็บบันทึกไฟล์ css javascripts
--> weights เก็บบันทึกไฟล์โมเดล
