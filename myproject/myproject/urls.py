"""myproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp.views import home, set100, djia, djiastock, djiastockpct, setstock, setstockpct


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home),
    path('home/', home),
    path('set100/', set100),
    path('djia/', djia),
    path('djia/price/<str:stock_n>/', djiastock),
    path('djia/pctchange/<str:stock_n>/', djiastockpct),
    path('set100/price/<str:stock_n>/', setstock),
    path('set100/pctchange/<str:stock_n>/', setstockpct),
]
