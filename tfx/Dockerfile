FROM tensorflow/serving:2.5.1

ARG MODEL_DIR
ARG MODEL_NAME

COPY --chown=tensorflow:tensorflow ./${MODEL_DIR}/${MODEL_NAME}_core /models/core/
COPY --chown=tensorflow:tensorflow ./${MODEL_DIR}/${MODEL_NAME}_diffnet /models/rank/
COPY --chown=tensorflow:tensorflow models.config /models/

CMD [ "--model_config_file=/models/models.config" ]
