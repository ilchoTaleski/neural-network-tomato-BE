
from rest_framework import serializers
from .models import TomatoNN

    
class TomatoSerializer(serializers.ModelSerializer):
    class Meta:
        model = TomatoNN
        fields = ('id', 'title')

