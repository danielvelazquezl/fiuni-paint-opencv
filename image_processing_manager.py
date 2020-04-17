import numpy as np
import cv2


def hex_to_rgb(value):
    """
    Conversor de un string hexadecimal a una tupla rgb.
    Fuente: https://www.codespeedy.com/convert-rgb-to-hex-color-code-in-python/
    """
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


class ImageProcessingManager:
    """
    Esta libreria contiene la implementación de los procesos internos del
    Editor de Imagenes.
    Desarrollador por: Federico Velazquez
    """

    DEFAULT_WIDTH = 512
    DEFAULT_HEIGHT = 512

    def __init__(self):
        super(ImageProcessingManager, self).__init__()

        # Por defecto tenemos una imagen blanca en la pila.
        initial_matrix = np.ones((self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT, 3), np.uint8) * 255

        # Estructura de imagenes
        self.stack_images = [initial_matrix]

        # Estructura de puntos/lineas
        self.stack_lines = []

    def last_image(self):
        """
        NO ALTERAR ESTA FUNCION
        Obtenemos la ultima imagen de nuestra estructura.
        """
        return self.stack_images[-1]

    def can_undo(self):
        """
        NO ALTERAR ESTA FUNCION
        Determinamos si la aplicación puede eliminar
        elementos de la pila.
        Debe haber por lo menos más de un elemento para que
        se pueda deshacer la imagen
        """
        return len(self.stack_images) > 1

    def has_changes(self):
        """
        NO ALTERAR ESTA FUNCION
        Determinamos si la aplicación contiene
        elementos de la pila.
        """
        return len(self.stack_images) > 1

    def add_image(self, image_path):
        """
        Leemos una imagen con OpenCV
        Redimensionamos segun los parametros: DEFAULT_WIDTH y DEFAULT_HEIGHT
        Agregamos una nueva imagen redimensionada en la pila.
        """
        self.stack_images.clear()
        img_resize = cv2.resize(cv2.imread(image_path, 0), (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT))
        self.stack_images.append(img_resize)

    def save_image(self, filename):
        """
        Guardamos la ultima imagen
        """
        cv2.imwrite(filename, self.last_image())

    def undo_changes(self):
        """
        Eliminamos el ultimo elemento guardado.
        """
        self.stack_images.pop()

    def save_points(self, x1, y1, x2, y2, line_width, color):
        """
        Guardamos informacion de los puntos aqui en self.stack_lines.
        """
        self.stack_lines.append((x1, y1, x2, y2, color, line_width))

    def add_lines_to_image(self):
        """
        Creamos una matriz, con un conjunto de lineas.
        Estas lineas se obtienen de self.stack_lines.

        Finalmente guardamos a nuestra pila de imagenes: self.stack_images.

        Ayuda: ver documentacion de "cv2.line" para dibujar lineas en una matriz
        Ayuda 2: no se olviden de limpiar self.stack_lines
        Ayuda 3: utilizar el metodo rgb_to_hex para convertir los colores
        """
        image = self.last_image().copy()
        for x1, y1, x2, y2, color, line_width in self.stack_lines:
            image = cv2.line(image, (x1, y1), (x2, y2), hex_to_rgb(color), int(line_width))

        self.stack_lines.clear()
        self.stack_images.append(image)

    def black_and_white_image(self):
        """
        Hacemos una copia de la ultima imagen.
        La Convertimos covertimos a blanco y negro.
        Guardamos a la estructura self.stack_images
        Retornamos la imagen procesada.
        """
        last = self.last_image().copy()
        bnw = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
        self.stack_images.append(bnw)
        return bnw

    def negative_image(self):
        """
        Hacemos una copia de la ultima imagen.
        Calculamos el negativo de la imagen.
        Guardamos a la estructura self.stack_images
        Retornamos la imagen procesada.
        """
        last = self.last_image().copy()
        negative = cv2.bitwise_not(last)
        self.stack_images.append(negative)
        return negative

    def global_equalization_image(self):
        """
        Hacemos una copia de la ultima imagen.
        Equalizamos la imagen.
        Guardamos a la estructura self.stack_images
        Retornamos la imagen procesada.
        """
        last = self.last_image().copy()
        eq = cv2.equalizeHist(last)
        self.stack_images.append(eq)
        return eq

    def CLAHE_equalization_image(self, grid=(8, 8), clip_limit=2.0):
        """
        Hacemos una copia de la ultima imagen.
        Equalizamos la imagen usando el algoritmo de CLAHE.
        Guardamos a la estructura self.stack_images
        Retornamos la imagen procesada.
        """
        last = self.last_image().copy()
        clahe = cv2.createCLAHE(clip_limit, grid)
        clahe_image = clahe.apply(last)
        self.stack_images.append(clahe_image)
        return clahe_image

    def contrast_and_brightness_processing_image(self, alpha, beta):
        """
        Hacemos una copia de la ultima imagen.
        Ajustamos la imagen segun parametros alpha y beta.
        Guardamos a la estructura self.stack_images
        Retornamos la imagen procesada.

        Fuente teorica: http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf
        Pagina 103

        OpenCV:
        https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html

        Función en OpenCV:
        https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#convertscaleabs
        """
        last = self.last_image().copy()
        img = cv2.convertScaleAbs(last, alpha, beta)
        self.stack_images.append(img)
        return img
