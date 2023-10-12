from rest_framework import serializers
from .models import MusicFile

class MusicFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = MusicFile
        fields = '__all__'
