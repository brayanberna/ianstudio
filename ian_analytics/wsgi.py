import os

from django.core.wsgi import get_wsgi_application

#os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ian_analytics.settings.local')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ian_analytics.settings.production')

#application = get_wsgi_application()

from dj_static import Cling
application = Cling(get_wsgi_application())
