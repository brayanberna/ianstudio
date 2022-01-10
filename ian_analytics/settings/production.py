from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['ianstudio.herokuapp.com']

# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

DATABASES = {
        'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'd1jdqg3cop169l',
        'USER': 'hvcyjcrcbzsqbl',
        'PASSWORD': '5ecbc329bfd9ec9902614ddcd56f3a6fb26e480fa8968c4069fe0a6fc6cafb5e',
        'HOST':'ec2-3-225-41-234.compute-1.amazonaws.com',
        'DATABASE_PORT':'5432',
   }
}