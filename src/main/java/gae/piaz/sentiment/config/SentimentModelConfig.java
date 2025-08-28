package gae.piaz.sentiment.config;

import java.io.IOException;
import java.nio.file.Paths;

import jakarta.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import ai.djl.ModelException;
import ai.djl.huggingface.translator.TextClassificationTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;

@Configuration
public class SentimentModelConfig {

    private static final Logger logger = LoggerFactory.getLogger(SentimentModelConfig.class);

    @Value("${sentiment.model-dir}")
    private String modelDir;

    private ZooModel<String, Classifications> model;

    @Bean
    public Predictor<String, Classifications> sentimentPredictor() {
        try {
            logger.info("Loading DistilRoBERTa financial news sentiment analysis model...");
            logger.info("Model directory: {}", modelDir);

            Criteria<String, Classifications> criteria = Criteria.builder()
                .setTypes(String.class, Classifications.class)
                .optModelPath(Paths.get(modelDir))
                .optModelName("model.pt")
                .optOption("modelDir", modelDir)
                .optTranslatorFactory(new TextClassificationTranslatorFactory())
                .optProgress(new ProgressBar())
                .build();

            model = criteria.loadModel();
            Predictor<String, Classifications> predictor = model.newPredictor();

            logger.info("Sentiment analysis model loaded successfully");
            return predictor;
        } catch (ModelException | IOException e) {
            logger.error("Error initializing sentiment analysis model", e);
            throw new RuntimeException("Failed to initialize sentiment analysis model \n" +
                "Make sure to run the python script convert_to_torchscript.py before starting the application", e);
        }
    }

    @PreDestroy
    public void cleanup() {
        try {
            if (model != null) {
                model.close();
            }
            logger.info("Sentiment analysis model resources released");
        } catch (Exception e) {
            logger.error("Error cleaning up sentiment analysis model resources", e);
        }
    }
}
