from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['ianstudio.herokuapp.com', '127.0.0.1']

# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

DATABASES = {
        'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'd9mq2a3vj3r4er',
        'USER': 'rwrxyffanfoqjl',
        'PASSWORD': 'c3fb620072ad28ae9f8fc8e99ed7d0b2a97e610ad40f13399c5271c8b82375ca',
        'HOST':'ec2-3-218-158-102.compute-1.amazonaws.com',
        'DATABASE_PORT':'5432',
   }
}