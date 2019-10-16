from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from .serializers import TomatoSerializer
from .models import TomatoNN
import json
from collections import namedtuple
from rest_framework import status
import webcolors
from django.http.multipartparser import MultiPartParser
from rest_framework.exceptions import ParseError
from rest_framework.parsers import FileUploadParser
from django.core.files.uploadedfile import InMemoryUploadedFile
from .forms import UploadFileForm
from PIL import Image
from io import BytesIO
from .neural import predictImage
from .neural import predictImages
from .neural import getFilters
from django.http import FileResponse
from django.http import HttpResponse
import base64
from io import BytesIO
from backend import urls

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


class TomatoView(viewsets.ModelViewSet):

    def create(self, request, *args, **kwargs):

        files = request.FILES['image']
        image = Image.open(files)

        predictions = predictImage(image)
        return Response(str(predictions[0][0]) + ',' + str(predictions[0][1]))


class TomatoViewAll(viewsets.ModelViewSet):

    def create(self, request, *args, **kwargs):

        files = request.FILES.getlist('image')
        images = []
        for x in files:
            images.append(Image.open(x))

        predictions = predictImages(images)
        return Response(predictions)

class TomatoFilters(viewsets.ModelViewSet):

    def create(self, request, *args, **kwargs):

        file = request.FILES['image']
        layer = request.data['layer']
        img = Image.open(file)

        filters = getFilters(img, int(layer))

        buffered_array = []

        for im in filters:
            buffered = BytesIO()
            im.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue())
            buffered_array.append(img_str)

        return Response(buffered_array)

class LayerCount(viewsets.ModelViewSet):

    def list(self, request, *args, **kwargs):
        model = urls.get_model()

        num = 0
        for x in model.layers:
            if 'conv' in x.name:
                num += 1

        return Response(num)