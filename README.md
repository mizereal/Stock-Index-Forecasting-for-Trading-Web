# Stock-Index-Forecasting-for-Trading-Web
เว็บแอปพลิเคชั่นสําหรับทํานายแนวโน้มราคาในตลาดตราสารทุน

myproject --> myapp --> 
- preprocessstock.py ดาวน์โหลดและจัดรูปข้อมูล
- predictor.py เรียกใช้งาน model
- runModel.py ฟังก์ชั่นสําหรับการทําการพยากรณ์ คัดเลือกและแบ่งแยกช่วงข้อมูลที่ต้องการ ส่งออกไป
- batch_manager.py จัดการ batch
- plot.py สร้างกราฟ
- pathModel.py เรียกชื่อเต็มของอัลกอริทึม
- views.py เรียกใช้งานฟังก์ชั่นทั้งหมด และส่งออกไปแสดงยังหน้าเว็บ

-- templates เก็บบันทึกไฟล์ html
-- static เก็บบันทึกไฟล์ css javascripts
-- weights เก็บบันทึกไฟล์โมเดล

# Folder --> myproject --> myapp --> requirements.txt
python library สําหรับติดตั้งที่ใช้ในโปรเจคนี้
