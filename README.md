## 1 домашнее задание

Выполнил ДЗ я один, так что на все роли можно написать - Азевич Марк.



## Для запуска нужно выполнить следующие шаги:
1. uvicorn main:app --reload
2. streamlit run dashboard.py



## Процесс обучения с REST построен следующим образом:
1. Запускается дэшборд streamlit
2. В соответствующей части выбирается нужная модель и прикладываются 3 .pkl файла: гиперпараметры, данные и таргет.
   
      Данные должна быть в формате list.
      Пример входных данных:
      [[5.4, 3.4, 1.7, 0.2],
       [4.9, 3.0, 1.4, 0.2],
       [6.3, 3.3, 6.0, 2.5]]

4. После добавления всех файлов и нажатия кнопки модель обучается и заносится в базу
5. Информацию по всем моделям можно посмотреть нажав List All Trained Models
6. При необходимости достать скоры можно указав id модели в соответствующей части. Скоры сохраняться в файле .pkl, название которого можно вписать ниже (например, scores.pkl)



**Процесс переобучения построен схожим образом**, отличие только в том, что класс модели и гиперпараметры не указываются а берутся из истории для модели с конкретным id.

**Всегдо доступны две модели: logistic regression и random forest.**

Модель включает логгирование, докстринги, также в /docs формируется автоматический сваггер. Есть отдельный эндпоинт для проверки статуса сервиса.

В папке test_datasets_and_hyps находятся тестовые данные: различные гиперпараметры для разных задач и два набора данных из sklearn.datasets.load_iris (тестировал я не только на них).

Также добавлен poetry.lock

