## Quickstart

1.  Prepare environment

    ```bash
    # Install python 3.9
    # Install docker version 20.10.17
    # Install docker-compose version v2.6.1
    pip install -r requirements.txt
    ```

2.  Prepare data

    -   Download data, `./data_warehouse/raw_data` dir should look like

        ```bash
        data_warehouse/raw_data
        ├── .gitkeep
        └── phase-1
            └── prob-1
                ├── features_config.json
                └── raw_train.parquet
        ```

    -   Process data

        ```bash
        python ./src/data_loader.py --phase-id phase-1 --prob-id prob-1
        python ./src/data_loader.py --phase-id phase-2 --prob-id prob-1
        python ./src/data_loader.py --phase-id phase-3 --prob-id prob-1
        ```

    -   After processing data, `./data_warehouse/processed_data` dir should look like

    ```bash
    data_warehouse/processed_data
    ├── .gitkeep
    └── phase-1
        └── prob-1
            ├── features_config.json
            ├── processed_data_x.parquet
            └── processed_data_y.parquet
    ```

3. Train model

    ```bash
    make mlflow_up
    export MLFLOW_TRACKING_URI=http://localhost:5040
    python src/model_trainer.py --phase-id phase-1 --prob-id prob-1 --name-run duplicated_drop_1
    python src/model_trainer.py --phase-id phase-2 --prob-id prob-1 --name-run duplicated_drop_1
    python src/model_trainer.py --phase-id phase-3 --prob-id prob-1 --name-run duplicated_drop_1
    ```

    -   Register model: - Go to mlflow UI at <http://localhost:5040> and 
                        - Register a new model named **phase1-prob1** & **phase1-prob2**
                        - Register a new model named **phase2-prob1** & **phase2-prob2**
                        - Register a new model named **phase3-prob1** & **phase3-prob2**

4.  Deploy model predictor

    -   Create model config at `src/config_files/model_config/phase-*/prob-*/model-1.yaml` with content: 

        ```yaml
        phase_id: "phase-1" OR "phase-2" OR "phase-3"
        prob_id: "prob-1" OR "prob-2"
        model_name: "phase1-prob1" OR "phase2-prob1" OR "phase3-prob1"
        model_version: "1"
        ```
    -   Test model predictor

        ```bash
        # run model predictor
        export MLFLOW_TRACKING_URI=http://localhost:5040
        python src/model_predictor.py --config-path src/config_files/model_config/phase-3/prob-1/model-1.yaml \
                                                    src/config_files/model_config/phase-3/prob-2/model-1.yaml \
                                      --port 8000

        # curl in another terminal
        curl -X POST http://localhost:8000/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data_warehouse/curl/phase-1/prob-1/payload-1.json

        # stop the predictor above
        ```

    -   Deploy model predictor
        
        **Note** Just use one of it, if want to use another, run ```make teardown``` then 
        - Not using nginx for load balancing
        - recheck class ProblemConst for ensure phase and prob id. (when phase change)
        - add new path to nginx config (new phase)

        ```bash
        export MLFLOW_TRACKING_URI=http://localhost:5000


        # those num worker set equal to num cpu processor

        uvicorn --workers 4

        # and change max_workers in concurrent.futures.ProcessPoolExecutor() predict function equal to workers uvicorn

        ```

        ```bash
        <!-- make mlflow_restart
        make predictor_up
        make predictor_curl -->
        ```
        - Using nginx
        ```
        make mlflow_restart
        make nginx_up
        make predictor_curl
        ```
        just down nginx and all of it predictor containers
        ```bash
        make nginx_down
        ```

    -   After running `make predictor_curl` or `make nginx_up` to send requests to the server, `./data_warehouse/captured_data` dir should look like:

        ```bash
         data_warehouse/captured_data
         ├── .gitkeep
         └── phase-1
             └── prob-1
                 ├── 123.parquet
                 └── 456.parquet
        ```
5.  Improve model

        ```bash
        python src/captured_data_processor.py --phase-id phase-2 --prob-id prob-1
        python src/captured_data_processor.py --phase-id phase-3 --prob-id prob-1

        python src/model_trainer.py --phase-id phase-2 --prob-id prob-1 --add-captured-data True
        python src/model_trainer.py --phase-id phase-3 --prob-id prob-1 --add-captured-data True
        ```
6.  Teardown

    ```bash
    make teardown
    ```