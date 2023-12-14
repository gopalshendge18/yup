from django.db import models

class CSVData(models.Model):
    csv_file = models.FileField(upload_to='csv_files/', null=True)
