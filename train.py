import tensorflow as tf
from transformers import TFXLMRobertaModel, AutoTokenizer, TFAutoModel
from datasets import load_dataset
from datetime import datetime
import logging
from pyprojroot.here import here
import os 

class mean_pooling_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(mean_pooling_layer, self).__init__()
    
    def call(self, inputs):
        token_embeddings = inputs[0]
        attention_mask = inputs[1]
        input_mask_expanded = tf.cast(
            tf.broadcast_to(tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)),
            tf.float32
        )

        embeddings = tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1) / tf.clip_by_value(tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)
        return embeddings
        

    def get_config(self):
        config = super(mean_pooling_layer, self).get_config()
        return config
        

def create_model():
    base_student_model = TFAutoModel.from_pretrained("distilbert-base-multilingual-cased",from_pt=True)
    input_ids_en = tf.keras.layers.Input(shape=(256,),name='input_ids_en', dtype=tf.int32)
    attention_mask_en = tf.keras.layers.Input(shape=(256,), name='attention_mask_en', dtype=tf.int32)
    input_ids_id = tf.keras.layers.Input(shape=(256,),name='input_ids_id', dtype=tf.int32)
    attention_mask_id = tf.keras.layers.Input(shape=(256,), name='attention_mask_id', dtype=tf.int32)
    mean_pooling = mean_pooling_layer()

    output_en = base_student_model.distilbert(input_ids_en, attention_mask=attention_mask_en).last_hidden_state[:,0,:]
    output_id = base_student_model.distilbert(input_ids_id, attention_mask=attention_mask_id).last_hidden_state[:,0,:]

    student_model = tf.keras.Model(inputs=[input_ids_en, attention_mask_en, input_ids_id, attention_mask_id], outputs=[output_en, output_id])
    print(student_model.summary())
    return student_model

class sentence_translation_metric(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        embeddings_en, embeddings_id = self.model.predict(val_dataset, verbose=1)
        # get the embeddings
        # compute the cosine similarity between the two 
        #normalize the embeddings
        similarity_matrix = tf.matmul(embeddings_en, embeddings_id, transpose_b=True)
        print(f"==>> similarity_matrix: {similarity_matrix}")
        # get the mean similarity
        correct_en_id = 0
        for i in range(similarity_matrix.shape[0]):
            if tf.math.argmax(similarity_matrix[i]) == i:
                correct_en_id += 1 

        similarity_matrix_T = tf.transpose(similarity_matrix)
        correct_id_en = 0
        for i in range(similarity_matrix_T.shape[0]):
            if tf.math.argmax(similarity_matrix_T[i]) == i:
                correct_id_en += 1

        acc_en_id = correct_en_id / similarity_matrix.shape[0]
        acc_id_en = correct_id_en / similarity_matrix_T.shape[0]
        avg_acc = (acc_en_id + acc_id_en) / 2
        print(f"translation accuracy from english to indonesian = {acc_en_id}")
        print(f"translation accuracy from indonesian to english = {acc_id_en}")
        print(f"average translation accuracy = {avg_acc}")

        logs["val_acc_en_id"] = acc_en_id
        logs["val_acc_id_en"] = acc_id_en
        logs["val_avg_acc"] = avg_acc


class ConstantScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, max_lr, warmup_steps=5000):
    super().__init__()
    self.max_lr = tf.cast(max_lr, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, tf.float32)
    condition  = tf.cond(step < self.warmup_steps, lambda: step / self.warmup_steps, lambda: 1.0)
    return self.max_lr * condition 
  

if __name__ == "__main__":
    num_data = 0
    dataset = load_dataset("carles-undergrad-thesis/en-id-parallel-sentences-embedding")
    
    dataset_1 = dataset["train"]

    # for split in dataset:
    #     dataset_1 = concatenate_datasets([dataset_1, dataset[split]])
    

    batch_size = 512
    dataset = dataset_1.train_test_split(test_size=0.01, shuffle=True)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    print(f"==>> val_dataset.shape: {val_dataset.shape}")
    
    train_dataset = train_dataset.to_tf_dataset(
        columns=["input_ids_en", "attention_mask_en", "input_ids_id", "attention_mask_id"],
        label_cols="target_embedding",
        batch_size=batch_size,
    ).unbatch()

    val_dataset = val_dataset.to_tf_dataset(
        columns=["input_ids_en", "attention_mask_en", "input_ids_id", "attention_mask_id"],
        label_cols="target_embedding",
        batch_size=batch_size,
    ).unbatch()

    #check feature
    print(train_dataset.element_spec)
    print(val_dataset.element_spec)

    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).cache()
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True).cache()


    warm_up_steps = 5_000_000 / batch_size *0.1
    learning_rate = ConstantScheduler(2e-5, warmup_steps= warm_up_steps)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)


    loss = tf.keras.losses.MeanSquaredError() 
    
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_path = here(f"disk/model/{date_time}/model.h5")

    if not os.path.exists(here("disk/model")):
        os.makedirs(here("disk/model"))

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath = output_path,
                    save_weights_only = True,
                    monitor = "val_avg_acc",
                    mode = 'auto',
                    verbose = 1,
                    save_best_only = True,
                    initial_value_threshold = 0.5,
                    )


    # tensor_board  = tf.keras.callbacks.TensorBoard(
    #                 log_dir = "gs://dicoding-capstone/output/logs/"+date_time
    # )

    if not os.path.exists(here("disk/performance_logs")):
        os.makedirs(here("disk/performance_logs"))
    

    csv_logger = tf.keras.callbacks.CSVLogger(
                    filename = here(f"disk/performance_logs/log-{date_time}.csv"),
                    separator = ",", 
                    append = False
    )

    
    callbacks = [sentence_translation_metric(), model_checkpoint, csv_logger]


    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver("local")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)

    with strategy.scope():
        student_model = create_model()
        student_model.compile(optimizer=optimizer, loss=loss) 

    student_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=callbacks)

    last_epoch_save = here(f"disk/model/last_epoch/{date_time}.h5")

    if not os.path.exists(here("disk/model/last_epoch")):
        os.makedirs(here("disk/model/last_epoch"))

    student_model.save_weights(last_epoch_save)
