"""
ASGI config for ian_analytics project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

#os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ian_analytics.settings.local')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ian_analytics.settings.production')

#application = get_asgi_application()

from dj_static import Cling
application = Cling(get_wsgi_application())
