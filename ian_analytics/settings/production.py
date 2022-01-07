from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = ['ianstudio.herokuapp.com']

# Database
# https://docs.djangoproject.com/en/3.1/ref/settings/#databases

DATABASES = {
        'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'd1r3rv9lktnrvn',
        'USER': 'npgdrafketfizt',
        'PASSWORD': 'aba7f21bd1c626f4de8cbd7836694a24f705f2140ea7fa9ac786478e7ffdcd4d',
        'HOST':'ec2-34-206-245-175.compute-1.amazonaws.com',
        'DATABASE_PORT':'5432',
   }
}