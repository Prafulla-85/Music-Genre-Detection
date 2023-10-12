from django.db import models

class MusicFile(models.Model):
    title = models.CharField(max_length=100)
    genre = models.CharField(max_length=100)
    audio_file = models.FileField(upload_to='audio_files/')
