from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['ianstudio.herokuapp.com']

# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

DATABASES = {
        'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'd5rmus0asf6m81',
        'USER': 'zsrsfxffzupcpw',
        'PASSWORD': '47023630f288bf04463e8ba0cbe5241ea0230e8f652038ede352647dcc99a661',
        'HOST':'ec2-35-168-73-79.compute-1.amazonaws.com',
        'DATABASE_PORT':'5432',
   }
}