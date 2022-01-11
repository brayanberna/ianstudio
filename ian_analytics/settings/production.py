from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['ianstudio.herokuapp.com']

# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

DATABASES = {
        'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'ddf59cdab0u9s0',
        'USER': 'qdwztjhexuwjat',
        'PASSWORD': 'ede754a33a952c763effa63f59fcb29014d9a0c4a552adcd7fd745b07606be10',
        'HOST':'ec2-54-211-74-66.compute-1.amazonaws.com',
        'DATABASE_PORT':'5432',
   }
}