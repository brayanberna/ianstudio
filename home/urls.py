from django.urls import path
from . import views

app_name = 'home'

urlpatterns = [
  path('task-list/', views.taskList, name='task-list'),
  path('task-detail/<str:pk>/', views.taskDetail, name='task-detail'),
  path('task-create/', views.taskCreate, name='task-create'),
  path('task-update/<str:pk>/', views.taskUpdate, name='task-update'),
  path('task-delete/<str:pk>/', views.taskDelete, name='task-delete'),
  path('detect_type_neuronal/', views.detect_type_neuronal, name='detect_type_neuronal'),
  path('run_red_neuronal/', views.run_red_neuronal, name='run_red_neuronal'),
  path('load_data/', views.load_data, name='load_data'),
  path('consult_red_neuronal/', views.consult_red_neuronal, name='consult_red_neuronal'),
]