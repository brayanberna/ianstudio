from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Database
# https://docs.djangoproject.com/en/3.2/ref/settings/#databases


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'ian_analytics',
        'USER': 'postgres',
        'PASSWORD': 'cindy057',
        'HOST':'127.0.0.1',
        'DATABASE_PORT':'5432',
   }
}