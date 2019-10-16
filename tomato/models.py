from django.db import models
import keras.models as km


class TomatoNN(models.Model):
    title = models.CharField(max_length=120)

    def _str_(self):
        return self.title

class CNNModel():
    def __init__(self, model=None, mobile_model=None):
        self.model = model
        self.mobile_model = mobile_model

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_mobile_model(self):
        return self.mobile_model

    def set_mobile_model(self, model):
        self.mobile_model = model

class SingletonModel(models.Model):

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        self.pk = 1
        super(SingletonModel, self).save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        pass

    @classmethod
    def load(cls):
        obj, created = cls.objects.get_or_create(pk=1)
        return obj

class SiteSettings(SingletonModel):
    default_model = km.Sequential()
    keras_model = models.Field(default=default_model)