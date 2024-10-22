import cv2
import numpy as np
import time
from collections import deque
import heapq
from ultralytics import YOLO
from client import trajectory_send
#from FIND_ANGLE import *
from undistort import Camera

model = YOLO(f"C:/Users/UrFU/Downloads/model17.pt")
COLOR = 'red' # red   green


def check_led_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    b_val = 8
    blur = cv2.medianBlur(hsv, 1 + b_val * 2)

    # create mask
    mask_green = cv2.inRange(blur, (0, 0, 0), (255, 116, 255))
    # морфология
    kernel_green = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel_green, iterations=3)
    # делатация
    dilation_green = cv2.dilate(mask_green, kernel_green, iterations=4)

    green_count = np.sum(dilation_green)


    mask_red = cv2.inRange(blur, (0,150,100), (255,255,255))
    # морфология
    kernel_red = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel_red, iterations=3)
    # делатация
    dilation_red = cv2.dilate(mask_red, kernel_red, iterations=4)

    red_count = np.sum(dilation_red)

    if green_count > red_count:
        return 'G'
    elif red_count > green_count:
        return 'R'
    else:
        return None

def check_self(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    b_val = 8
    blur = cv2.medianBlur(hsv, 1 + b_val * 2)

    # create mask
    mask_green = cv2.inRange(blur, (0, 0, 170), (255, 116, 255))
    
    green_count = np.sum(mask_green)
    print(green_count)

    return green_count


def draw_boxes(frame, results, binary_image):
    green_robot_centers = []  # Список для центров зеленых роботов
    red_robot_centers = []    # Список для центров красных роботов
    cubes = []
    # base = []
    # buttons = []

    for result in results:
        boxes = result.boxes
        global our_robot_center
        green_count_max = 0
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            conf = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]



            if (x2 - x1) * (y2 - y1) < 18000: #and round(float(conf), 2) > 0.4:
                label = f'{class_name} {conf:.2f}'
                if class_name == 'robot':
                    # Определение цвета LED
                    led_color = check_led_color(frame[y1:y2,x1:x2])
                    green_count = check_self(frame[y1:y2,x1:x2])

                    # Изменение метки в зависимости от цвета
                    if led_color == 'G':
                        label = f'G {class_name} {conf:.2f}'
                        # Сохраняем центр зеленого робота
                        green_robot_centers.append((center_x, center_y))
                        if green_count > green_count_max and COLOR == 'green':
                            green_count_max = green_count
                            our_robot_center = (center_x, center_y)
                    elif led_color == 'R':
                        label = f'R {class_name} {conf:.2f}'
                        # Сохраняем центр красного робота
                        red_robot_centers.append((center_x, center_y))
                        if green_count > green_count_max and COLOR == 'red':
                            green_count_max = green_count
                            our_robot_center = (center_x, center_y)
                    
                    cv2.rectangle(binary_image, (x1, y1), (x2, y2), 0, -1)  # Черный цвет (0), закрашиваем (-1)
                    # cv2.imshow('Detection', binary_image)    
                elif class_name == 'cube':
                    cubes.append((center_x, center_y))

                # Рисуем прямоугольник и текст на оригинальном изображении
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (255, 0, 255), -1)  # Цвет: фиолетовый (BGR: 255, 0, 255)

    # Возвращаем списки центров зеленых и красных роботов
    return binary_image, green_robot_centers, red_robot_centers


def draw_edge_center_lines(output_image, approx):
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(approx)
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # Convert to integer values

    # Calculate midpoints of each edge of the rectangle
    midpoints = []
    for i in range(4):
        pt1 = box[i]
        pt2 = box[(i + 1) % 4]
        midpoint = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        midpoints.append(midpoint)

    # Draw lines passing through the centers of opposite edges
    cv2.line(output_image, midpoints[0], midpoints[2], (0, 0, 255), 2)  # Horizontal line
    cv2.line(output_image, midpoints[1], midpoints[3], (0, 0, 255), 2)  # Vertical line



def draw_center_marker(output_image):
    # Get the image dimensions
    height, width, _ = output_image.shape
    center_x = width // 2
    center_y = height // 2

    # Define the points of the diamond shape (rhombus)
    diamond_size = 10  # Adjust size as needed
    points = np.array([
        [center_x, center_y - diamond_size],  # Top
        [center_x + diamond_size, center_y],  # Right
        [center_x, center_y + diamond_size],  # Bottom
        [center_x - diamond_size, center_y]   # Left
    ])

    # Draw the pink rhombus
    cv2.polylines(output_image, [points], isClosed=True, color=(255, 105, 180), thickness=3)



def draw_bounding_rotated_rectangle(output_image, color=(0, 255, 255)):  # Yellow color
    # Invert the binary image to detect the central white area
    inverted_binary = cv2.bitwise_not(binary_image)

    # Find contours on the inverted binary image
    contours_inverted, _ = cv2.findContours(inverted_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour represents the central white area
    largest_contour = max(contours_inverted, key=cv2.contourArea)

    # Get the minimum area rectangle that bounds the central white area
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # Convert to integer values

    # Draw the bounding rotated rectangle with the specified color
    cv2.drawContours(output_image, [box], 0, color, 2)



def find_point_on_rect_side(vertices, side_index, offset_along, offset_perpendicular):
    """
    Находит координаты точки на стороне повёрнутого прямоугольника со смещением по перпендикуляру.
    
    :param vertices: Список из 4 вершин повёрнутого прямоугольника, каждая вершина представлена кортежем (x, y).
    :param side_index: Индекс стороны прямоугольника (0 - первая сторона, 1 - вторая и т.д.).
    :param offset_along: Смещение вдоль выбранной стороны в пикселях.
    :param offset_perpendicular: Смещение по перпендикуляру относительно стороны в пикселях.
    :return: Координаты точки на стороне (x, y).
    """
    # Стороны прямоугольника определяются вершинами (в порядке обхода)
    side_start = vertices[side_index]
    side_end = vertices[(side_index + 1) % 4]  # Используем модуль для циклического доступа к вершинам

    # Вычисляем вектор стороны
    side_vector = np.array(side_end) - np.array(side_start)
    side_length = np.linalg.norm(side_vector)

    # if offset_along > side_length:
    #     raise ValueError("Смещение вдоль стороны превышает длину стороны.")

    # Нормализуем вектор стороны
    side_unit_vector = side_vector / side_length

    # Вектор перпендикуляра (поворачиваем вектор стороны на 90 градусов)
    perpendicular_vector = np.array([-side_unit_vector[1], side_unit_vector[0]])

    # Находим точку на стороне
    point_along_side = np.array(side_start) + side_unit_vector * offset_along

    # Смещаем точку по перпендикуляру
    point = point_along_side + perpendicular_vector * offset_perpendicular

    return point

# Функция для поиска границ связной белой области вокруг данной точки
def find_boundaries(image, start, include_diagonals=False):
    rows, cols = image.shape
    x_start, y_start = start

    if image[x_start, y_start] != 255:
        return None

    queue = deque([(x_start, y_start)])
    visited = set([(x_start, y_start)])

    min_x = max_x = x_start
    min_y = max_y = y_start

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if include_diagonals:
        directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    while queue:
        x, y = queue.popleft()

        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy

            if 0 <= new_x < rows and 0 <= new_y < cols and (new_x, new_y) not in visited and image[new_x, new_y] == 255:
                queue.append((new_x, new_y))
                visited.add((new_x, new_y))

    return min_x, max_x, min_y, max_y

# Закрашиваем все, кроме найденной области
def mask_outside_region(binary_image, boundaries):
    min_x, max_x, min_y, max_y = boundaries
    mask = np.zeros_like(binary_image)

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if binary_image[x, y] == 255:
                mask[x, y] = 255

    return mask

# Находим минимальный повёрнутый прямоугольник
def find_min_rotated_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return box

# Function to draw the minimum area rectangle around the inner white area
def draw_rotated_bounding_box(output_image, inner_contour):
    # Get the minimum area rectangle for the inner white region
    rect = cv2.minAreaRect(inner_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # Convert to integer values

    # Draw the rectangle on the output image
    cv2.drawContours(output_image, [box], 0, (255, 0, 0), 2)

def calculate_avg_distance_to_center(rect, image_center):
    # Get the box points of the rectangle
    box = cv2.boxPoints(rect)

    # Calculate the average distance of the box points to the center of the image
    distances = [np.linalg.norm(np.array(point) - np.array(image_center)) for point in box]
    return np.mean(distances)

# A* algorithm implementation with full radius check for the robot
def heuristic(a, b):
    # return np.linalg.norm(np.array(a) - np.array(b))
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* с проверкой радиуса и поворотами на 90 градусов
# A* с проверкой радиуса и поворотами на 90 и 45 градусов
def astar_with_full_radius_check(grid, start, goal, robot_radius, include_diagonals=False):
    # Восемь направлений для движения: вверх, вниз, влево, вправо, и по диагоналям
    directions =[(0, 1), (0, -1), (1, 0), (-1, 0)]
    if include_diagonals:
        directions+=[(1, 1), (1, -1), (-1, 1), (-1, -1)]

    # Множества для отслеживания посещенных и будущих узлов
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}

    # Отслеживание предыдущего направления для поворотов
    previous_direction = {start: None}

    open_set = []
    heapq.heappush(open_set, (fscore[start], start))

    while open_set:
        current = heapq.heappop(open_set)[1]

        # Если достигли цели, восстанавливаем путь
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        close_set.add(current)
        for direction in directions:
            neighbor = current[0] + direction[0], current[1] + direction[1]

            # Проверка, может ли робот с радиусом двигаться на эту клетку
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                min_y = max(0, neighbor[0] - robot_radius)
                max_y = min(grid.shape[0], neighbor[0] + robot_radius + 1)
                min_x = max(0, neighbor[1] - robot_radius)
                max_x = min(grid.shape[1], neighbor[1] + robot_radius + 1)

                # Если робот не может поместиться, пропускаем
                if np.any(grid[min_y:max_y, min_x:max_x] == 255):
                    continue
            else:
                continue

            tentative_g_score = gscore[current] + heuristic(current, neighbor)

            # Учет поворотов: если направление изменилось — добавляем штраф
            if previous_direction[current] is not None and previous_direction[current] != direction:
                # Если поворот на 90 градусов (вверх, вниз, влево, вправо)
                if abs(direction[0]) + abs(direction[1]) == 1:
                    tentative_g_score += 1  # Штраф за 90 градусов


            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in open_set]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                # Сохраняем текущее направление для вычисления поворотов
                previous_direction[neighbor] = direction

                heapq.heappush(open_set, (fscore[neighbor], neighbor))

    return False
# Function to detect straight segments and turning points
def detect_segments_and_turns(path):
    segments = []
    turning_points = []
    current_direction = None

    for i in range(1, len(path)):
        y_diff = path[i][0] - path[i - 1][0]
        x_diff = path[i][1] - path[i - 1][1]
        direction = (y_diff, x_diff)

        if direction != current_direction:
            if current_direction is not None:
                # Save the previous segment and mark turning point
                segments.append((path[start_idx], path[i - 1]))
                turning_points.append(path[i - 1])
            # Start a new segment
            current_direction = direction
            start_idx = i - 1

    # Add the last segment
    segments.append((path[start_idx], path[-1]))

    return segments, turning_points

# Visualization of the path with segments and turning points
def visualize_segments_and_turns(image, segments, turning_points, block_size):
    # Draw segments
    start = 0
    end = 0
    start = segments[0][0]
    points = []
    points.append([start[1] * block_size + block_size // 2, start[0] * block_size + block_size // 2])
    for segment in segments:
        start, end = segment
        start_point = (start[1] * block_size + block_size // 2, start[0] * block_size + block_size // 2)
        end_point = (end[1] * block_size + block_size // 2, end[0] * block_size + block_size // 2)
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    # Mark turning points
    for point in turning_points:
        center = [point[1] * block_size + block_size // 2, point[0] * block_size + block_size // 2]
        points.append(center)
        cv2.circle(image, center, 5, (0, 0, 255), -1)
    point_end = [points[-1][0] -  (end[1] * block_size + block_size // 2), points[-1][1] - (end[0] * block_size + block_size // 2)]
    print(points[-1][1])
    print((end[1] * block_size + block_size // 2))
    print(points[-1][0])
    print((end[0] * block_size + block_size // 2))
    print (point_end)
    offset = 5
    if point_end[0] != 0:
        if point_end[0]>0:
            points.append([(end[1]+offset) * block_size + block_size // 2, end[0]* block_size + block_size // 2])
        else:
            points.append([(end[1]-offset) * block_size + block_size // 2, end[0]* block_size + block_size // 2])
    elif point_end[1] !=0:
        if point_end[1]>0:
            points.append([end[1] * block_size + block_size // 2, (end[0] +offset)* block_size + block_size // 2])
        else:
            points.append([end[1] * block_size + block_size // 2, (end[0] -offset)* block_size + block_size // 2])
    
    return points

def init_postprocess(results, binary_image):

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            # conf = box.conf[0]  
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if (x2 - x1) * (y2 - y1) < 18000: #and round(float(conf), 2) > 0.4:
                if class_name in ['robot', 'cube', 'ball']:
                    # Закрашиваем bounding box на бинарном изображении черным цветом
                    # print(x1, y1, x2, y2)
                    cv2.rectangle(binary_image, (x1, y1), (x2, y2), 0, -1)  # Черный цвет (0), закрашиваем (-1)
                    # cv2.imshow('Detection', binary_image)    
                elif class_name in ['OG_btn', 'BP_btn']:
                    cv2.rectangle(binary_image, (x1, y1), (x2, y2), 255, -1)  # Черный цвет (0), закрашиваем (-1)

                if class_name == 'cube':
                    cubes.append((center_x, center_y))
                elif class_name == 'OG_btn' or class_name == 'BP_btn':
                    buttons.append((center_x, center_y))
                elif class_name == 'green_base' and COLOR == 'green' or class_name == 'red_base' and COLOR == 'red':
                    base.append((center_x, center_y))
                
    # Возвращаем списки центров зеленых и красных роботов
    return binary_image


def init(frame):

    frame_undistorted = camera.undistort_and_crop_frame(frame)

    # Преобразование в цветовое пространство LAB
    lab_image = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2Lab)

    # Пороговые значения для каналов LAB
    l_max = 95

    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Бинаризация канала L: темные области станут белыми
    _, l_binary = cv2.threshold(l_channel, l_max, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(l_binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    erosion_dilation_iterations = 1

    kernel = np.ones((3, 3), np.uint8)  # Настраиваем ядро эрозии

    # Применяем эрозию
    binary_image = cv2.erode(binary_image, kernel, iterations=erosion_dilation_iterations)

    # Применяем дилатацию
    binary_image = cv2.dilate(binary_image, kernel, iterations=erosion_dilation_iterations)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the center of the image
    image_center = (binary_image.shape[1] // 2, binary_image.shape[0] // 2)

    # Параметры области и начальной точки
    start_point = (image_center[1], image_center[0])
    boundaries = find_boundaries(binary_image, start_point)

    # Маскируем изображение
    masked_image = mask_outside_region(binary_image, boundaries)

    # Поиск контуров на изображении
    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Находим и рисуем минимальный повёрнутый прямоугольник
    output_image_closest_rect = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # Создаем маску, соответствующую размеру исходного изображения
    central_mask = np.zeros(output_image_closest_rect.shape[:2], dtype=np.uint8)

    if contours:

        contour = contours[0]
        min_rotated_rect = find_min_rotated_rect(contour)

        # Создаем маску, соответствующую размеру исходного изображения
        mask = np.zeros(output_image_closest_rect.shape[:2], dtype=np.uint8)

        # Заполняем область прямоугольника белым на маске
        cv2.drawContours(mask, [min_rotated_rect], 0, 255, -1)
        cv2.drawContours(central_mask, [min_rotated_rect], 0, 0, -1)

        offset = 10

        # Выполняем эрозию области на маске
        kernel = np.ones((offset*2, offset*2), np.uint8)  # Настраиваем ядро эрозии
        eroded_mask = cv2.erode(mask, kernel, iterations=1)

        # Инвертируем обработанную маску для заполнения черным внутренней области
        inner_mask = cv2.bitwise_not(eroded_mask)

        # Внутреннюю часть (где эрозия) заполним черным
        output_image_closest_rect[inner_mask == 0] = [0, 0, 0]

        # Рисуем контур прямоугольника
        cv2.drawContours(output_image_closest_rect, [min_rotated_rect], 0, (0, 0, 255), 2)

        korner_offset = 60

        white_pixel_count = []
        black_pixel_count = []

        for i in range(4):
            points = np.array([find_point_on_rect_side(min_rotated_rect, i, korner_offset, offset),
                    find_point_on_rect_side(min_rotated_rect, i, korner_offset, 0),
                    find_point_on_rect_side(min_rotated_rect, (i+1)%4, 0, korner_offset),
                    find_point_on_rect_side(min_rotated_rect, (i+1)%4, offset, korner_offset)])

            door_rect = [points.astype(int)]
            cv2.drawContours(output_image_closest_rect, door_rect, -1, (0, 255, 0), 2)

            # Создаем маску такого же размера, как и бинарное изображение
            mask = np.zeros_like(binary_image)

            # Заполняем маску в области прямоугольника белым цветом (1)
            cv2.drawContours(mask, door_rect, -1, (255), thickness=cv2.FILLED)

            # Применяем маску к бинарному изображению
            masked_image = cv2.bitwise_and(binary_image, binary_image, mask=mask)

            # Считаем количество белых и черных пикселей в области прямоугольника
            total_pixels_in_rect = np.sum(mask == 255)  # Общее количество пикселей в прямоугольнике
            white_pixels = np.sum(masked_image == 255)  # Белые пиксели (единицы в бинарном изображении)
            black_pixels = total_pixels_in_rect - white_pixels  # Остальные будут черными

            # Подсчитываем количество белых пикселей (это и будут пиксели контуров)
            white_pixel_count.append(white_pixels)
            black_pixel_count.append(black_pixels)

        small_offset = -5
        korner_offset_central_mask = 25
        if black_pixel_count[i] + black_pixel_count[(i+2)%4] >= black_pixel_count[(i+1)%4] + black_pixel_count[(i+3)%4] :
            points = np.array([find_point_on_rect_side(min_rotated_rect, 0, small_offset, korner_offset_central_mask),
                find_point_on_rect_side(min_rotated_rect, 1, korner_offset_central_mask, small_offset),
                find_point_on_rect_side(min_rotated_rect, 2, small_offset, korner_offset_central_mask),
                find_point_on_rect_side(min_rotated_rect, 3, korner_offset_central_mask, small_offset)])
        else:
            points = np.array([find_point_on_rect_side(min_rotated_rect, 0, korner_offset_central_mask, small_offset),
                find_point_on_rect_side(min_rotated_rect, 1, small_offset, korner_offset_central_mask),
                find_point_on_rect_side(min_rotated_rect, 2, korner_offset_central_mask, small_offset),
                find_point_on_rect_side(min_rotated_rect, 3, small_offset, korner_offset_central_mask)])


        corner_rect = [points.astype(int)]

        # Заполняем маску в области прямоугольника белым цветом (1)
        cv2.drawContours(central_mask, corner_rect, -1, (255), thickness=cv2.FILLED)

        central_mask = cv2.bitwise_not(central_mask)

        # Применяем маску к бинарному изображению
        binary_image = cv2.bitwise_and(binary_image, binary_image, mask=central_mask)

        # cv2.imshow('BINARY', binary_image)
        central_mask_xy = [(min_rotated_rect[0][0] + min_rotated_rect[2][0])//2, (min_rotated_rect[0][1] + min_rotated_rect[2][1])//2]

    else:
        raise NameError('Контуры не найдены')

    frame_undistorted = crop_field(frame_undistorted, central_mask_xy)
    binary_image = crop_field(binary_image, central_mask_xy)
    results = model(frame_undistorted,
                    save=False,
                    imgsz=576,
                    conf=0.1,
                    iou = 0.6,
                    max_det=15)
    init_postprocess(results, binary_image)

    # binary_image = binary_image[0:frame_height, central_mask_xy[0]-field_offset:central_mask_xy[0]+field_offset]

    width = int(binary_image.shape[1] * resolution_scale)
    height = int(binary_image.shape[0] * resolution_scale)

    binary_image = cv2.resize(binary_image, (width, height), interpolation=cv2.INTER_AREA)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # Get the center of the image
    image_center = (binary_image.shape[1] // 2, binary_image.shape[0] // 2)


    # Block size
    block_size = int(10 * resolution_scale)

    # Convert the image into a grid for pathfinding
    height, width = binary_image.shape
    grid_height, grid_width = height // block_size, width // block_size
    grid = np.zeros((grid_height, grid_width), dtype=int)

    # Fill the grid with 0 (walkable) or 255 (non-walkable)
    for y in range(0, grid_height):
        for x in range(0, grid_width):
            block = binary_image[y*block_size:(y+1)*block_size, x*block_size:(x+1)*block_size]
            if np.all(block == 255):  # White - obstacle
                grid[y, x] = 255  # Obstacle

    cv2.imwrite('test_map.png', binary_image)

    init_map = binary_image
    init_grid = grid

    return central_mask, central_mask_xy, init_map, init_grid



def crop_field(image, center_xy):

    image = image[0:frame_height, center_xy[0]-field_offset:center_xy[0]+field_offset]
    return image

def rdp(points, epsilon):
    """
    Apply the Ramer-Douglas-Peucker algorithm to smooth a given path.
    points: List of tuples (x, y)
    epsilon: The distance threshold for smoothing
    """
    if len(points) < 3:
        return points

    # Find the point with the maximum distance from the line between start and end
    start = np.array(points[0])
    end = np.array(points[-1])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    line_unit_vec = line_vec / line_len if line_len != 0 else np.zeros_like(line_vec)

    dmax = 0
    index = 0
    for i, point in enumerate(points[1:-1], 1):
        point_vec = np.array(point) - start
        proj_len = np.dot(point_vec, line_unit_vec)
        proj_point = start + proj_len * line_unit_vec
        dist = np.linalg.norm(np.array(point) - proj_point)
        if dist > dmax:
            index = i
            dmax = dist

    # If the maximum distance is greater than the threshold, recursively simplify
    if dmax > epsilon:
        left = rdp(points[:index + 1], epsilon)
        right = rdp(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]

def get_path(points):
    Kx = 2.05
    Ky = 2.5
    res = np.array(points)
    res_list = [[int(y/Ky), int(x/Kx)] for x, y in res]
    res = [list(ele) for ele in res_list]
    return res


def adjust_goal_to_stop_before(goal_pixel, current_position, stop_distance_cm, resolution_scale, block_size):
    # Рассчитать расстояние в пикселях для остановки
    stop_distance_pixels = stop_distance_cm / resolution_scale

    # Рассчитать разницу между текущей позицией и целью
    delta_x = goal_pixel[0] - current_position[0]
    delta_y = goal_pixel[1] - current_position[1]

    # Рассчитать общее расстояние от робота до цели
    distance_to_goal = np.abs(delta_x) + np.abs(delta_y)

    # Если робот уже находится достаточно близко, оставить цель без изменений
    if distance_to_goal <= stop_distance_pixels:
        return goal_pixel

    # Найти новую цель на линии между роботом и исходной целью, на расстоянии stop_distance_pixels от цели
    new_goal_x = goal_pixel[0] - (delta_x * stop_distance_pixels / distance_to_goal)
    new_goal_y = goal_pixel[1] - (delta_y * stop_distance_pixels / distance_to_goal)

    return [int(new_goal_x), int(new_goal_y)]

def send_flag(center_flag, cube_flag):
    return  [center_flag, cube_flag]
# Открываем видео

# cap = cv2.VideoCapture('output_right_datas.avi')

ip_camera_url_right = "rtsp://Admin:rtf123@192.168.2.250/251:554/1/1"
cap = cv2.VideoCapture(ip_camera_url_right)
# cap = cv2.VideoCapture('C:/Users/UrFU/Downloads/test1.avi')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

camera = Camera('left', frame_size)
# Camera('right', frame_size)

# Variables for calculating FPS
prev_time = 0

resolution_scale = 0.8

# Проверяем, открылось ли видео
if not cap.isOpened():
    print("Ошибка открытия видео")
    exit()

cubes = []
buttons = []
base = []

our_robot_center = 0

ret, frame = cap.read()
field_offset = 565
central_mask, central_mask_xy, init_map, init_grid = init(frame)
central_mask = crop_field(central_mask, central_mask_xy)

cv2.namedWindow('YOLOOOOOOOOv8n Detection', cv2.WINDOW_NORMAL)  # Установить окно с возможностью изменения размеров
cv2.resizeWindow('YOLOOOOOOOOv8n Detection', int(field_offset*2*1280/1920), 720)  # Изменить размер окна до 800x600

cv2.namedWindow('Path', cv2.WINDOW_NORMAL)  # Установить окно с возможностью изменения размеров
cv2.resizeWindow('Path', int(field_offset*2*1280/1920), 720)  # Изменить размер окна до 800x600

# Обработка каждого кадра
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break  # Если кадры закончились

    # Get the current time and calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Convert FPS to integer for display
    fps_text = f"FPS: {round(fps,2)}"

    frame_undistorted = camera.undistort_and_crop_frame(frame)

    frame_undistorted = crop_field(frame_undistorted, central_mask_xy)
    # cv2.imshow("CROP", frame_undistorted)
    # Преобразование в цветовое пространство LAB
    lab_image = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2Lab)

    # Пороговые значения для каналов LAB
    l_max = 95

    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Бинаризация канала L: темные области станут белыми
    _, l_binary = cv2.threshold(l_channel, l_max, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(l_binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    erosion_dilation_iterations = 1

    kernel = np.ones((3, 3), np.uint8)

    # Применяем эрозию
    binary_image = cv2.erode(binary_image, kernel, iterations=erosion_dilation_iterations)

    # Применяем дилатацию
    binary_image = cv2.dilate(binary_image, kernel, iterations=erosion_dilation_iterations)

    binary_image = cv2.bitwise_and(binary_image, binary_image, mask=central_mask)

    results = model(frame_undistorted,
                    save=False,
                    imgsz=576,
                    conf=0.2,
                    iou = 0.6,
                    max_det=15)
    LAB = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    binary_image, green_robot_centers, red_robot_centers = draw_boxes(frame_undistorted, results, binary_image)

    cv2.putText(frame_undistorted, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('YOLOOOOOOOOv8n Detection', frame_undistorted)
    # except Exception as err:
    #     print(f"Unexpected {err=}, {type(err)=}")
    #     raise

    width = int(binary_image.shape[1] * resolution_scale)
    height = int(binary_image.shape[0] * resolution_scale)
    binary_image = cv2.resize(binary_image, (width, height), interpolation=cv2.INTER_AREA)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # Get the center of the image
    image_center = (binary_image.shape[1] // 2, binary_image.shape[0] // 2)


    # Block size
    block_size = int(10 * resolution_scale)

    # print (central_mask_xy)
    # Define start and goal points in pixel coordinates
    if COLOR == 'red':
        start_pixel = [int(red_robot_centers[0][1] * resolution_scale),
                       int(red_robot_centers[0][0] * resolution_scale)]  # Example start pixel coordinate (y, x)
    else:
        start_pixel = [int(green_robot_centers[0][1] * resolution_scale),
                       int(green_robot_centers[0][0] * resolution_scale)]
    # print(our_robot_center)
    # start_pixel = [int(our_robot_center[1] * resolution_scale), int(our_robot_center[0] * resolution_scale)]
    goal_center_pixel =(int(central_mask_xy[1] * resolution_scale), width//2)

        

    goal_cube_pixel = [int(cubes[0][1] * resolution_scale), int(cubes[0][0] * resolution_scale)]

    goal_pixel = (400, 700)  # Example goal pixel coordinate (y, x)

    # Convert pixel coordinates to grid coordinates
    start = (start_pixel[0] // block_size, start_pixel[1] // block_size)
    goal_center = (goal_center_pixel[0] // block_size, goal_center_pixel[1] // block_size)
    goal_cube = (goal_cube_pixel[0] // block_size, goal_cube_pixel[1] // block_size)

    # Define robot's radius (in terms of grid cells)
    robot_radius = int(42 * resolution_scale // block_size)  # Convert radius from pixels to grid cells

    # Find the path using the modified A* with full radius check
    path_to_center = astar_with_full_radius_check(init_grid, start, goal_center, robot_radius)
    path_to_cube = astar_with_full_radius_check(init_grid, start, goal_cube, robot_radius)
    center_flag = False
    cube_flag = False
    # if path_to_center and path_to_cube:
    #     if len(path_to_center) <= len(path_to_cube):
    #         chosen_path = path_to_center
    #         chosen_goal = goal_center_pixel
    #         center_flag = True
    #     else:
    #         chosen_path = path_to_cube
    #         chosen_goal = goal_cube_pixel
    #         cube_flag = True
    # elif path_to_center:
    #     chosen_path = path_to_center
    #     chosen_goal = goal_center_pixel
    #     center_flag = True
    # elif path_to_cube:
    #     chosen_path = path_to_cube
    #     chosen_goal = goal_cube_pixel
    #     cube_flag = True
    # else:
    #     chosen_path = None
    output_image_color = cv2.cvtColor(init_map, cv2.COLOR_GRAY2BGR)
    if COLOR == 'red':

        cv2.circle(output_image_color,
                   (int(red_robot_centers[0][0] * resolution_scale), int(red_robot_centers[0][1] * resolution_scale)),
                   5, (0, 255, 255), -1)
    else:
        cv2.circle(output_image_color,
                   (int(green_robot_centers[0][0] * resolution_scale),
                    int(green_robot_centers[0][1] * resolution_scale)),
                   5, (0, 255, 255), -1)
    # cv2.circle(output_image_color, (358, 452) , 5, (255, 0, 255), -1)

    cv2.circle(output_image_color, (int(cubes[0][0] * resolution_scale), int(cubes[0][1] * resolution_scale)), 5,
               (255, 0, 255), -1)

    # If a path is found, visualize it with segments and turning points
    center_flag = True
    if path_to_center:
        segments, turning_points = detect_segments_and_turns(path_to_center)

        # points.append([int(cubes[0][0]* resolution_scale), int(cubes[0][1]* resolution_scale)])
        traj = get_path(visualize_segments_and_turns(output_image_color, segments, turning_points, block_size))
        print(traj)
        # send_flag(center_flag, cube_flag)
        if center_flag:
            trajectory_send(traj, 'ball')
        else:
            trajectory_send(traj, 'cube')
            

        cv2.imshow('Path', output_image_color)
        cv2.waitKey(10000)
        break


    else:

        cv2.putText(output_image_color, "Path not found",
                    (int(image_center[0] * resolution_scale), int(image_center[1] * resolution_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Path', output_image_color)

    # if COLOR == 'red':
    #     new_start_pixel = [int(red_robot_centers[0][1] * resolution_scale), int(red_robot_centers[0][0] * resolution_scale)]  # Example start pixel coordinate (y, x)
    # else:
    #     new_start_pixel = [int(green_robot_centers[0][1] * resolution_scale), int(green_robot_centers[0][0] * resolution_scale)]
    # #new_start_pixel = [int(cubes[0][1] * resolution_scale), int(cubes[0][0] * resolution_scale)]
    #
    # new_goal_pixel = [int(base[0][1] * resolution_scale), int(base[0][0] * resolution_scale)]
    # new_start = (new_start_pixel[0] // block_size, new_start_pixel[1] // block_size)
    # print(new_start)
    # new_goal = (new_goal_pixel[0] // block_size, new_goal_pixel[1] // block_size)
    # robot_radius = int(40 * resolution_scale // block_size)
    # path = astar_with_full_radius_check(init_grid, new_start, new_goal, robot_radius)
    # print(new_goal)
    # output_image_color = cv2.cvtColor(init_map, cv2.COLOR_GRAY2BGR)
    # if path:
    #     segments, turning_points = detect_segments_and_turns(path)
    #
    #     # points.append([int(cubes[0][0]* resolution_scale), int(cubes[0][1]* resolution_scale)])
    #     traj = get_path(visualize_segments_and_turns(output_image_color, segments, turning_points, block_size))
    #     print(traj)
    #     #trajectory_send(traj)
    #     cv2.imshow('Path', output_image_color)
    #     cv2.waitKey(10000)
    #
    # else:
    #
    #     cv2.putText(output_image_color, "Path not found", (int(image_center[0] * resolution_scale), int(image_center[1] * resolution_scale)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #     cv2.imshow('Path', output_image_color)
    #     break
    # # Put the FPS text on the frame
    # cv2.putText(output_image_with_marker, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # cv2.imshow('Frame', output_image_with_marker)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print ('EXIT')
        break



# Освобождаем захват видео
cap.release()