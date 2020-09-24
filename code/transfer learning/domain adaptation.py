from task.task import ObjectRecognitionTask, DomainAdaptationTask, SubcategoryRecognitionTask, \
    SceneObjectRecognitionTask

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    object_recognition_task.test()

    # DOMAIN ADAPTATION - OFFICE
    domain_adaptation_task = DomainAdaptationTask(origin_domain="amazon", target_domain="webcam", combo="S")
