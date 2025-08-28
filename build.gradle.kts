plugins {
	java
	id("org.springframework.boot") version "3.4.5"
	id("io.spring.dependency-management") version "1.1.7"
}

group = "gae.piaz"
version = "0.0.1-SNAPSHOT"

java {
	toolchain {
		languageVersion = JavaLanguageVersion.of(24)
	}
}

repositories {
	mavenCentral()
}

dependencies {
	compileOnly("org.projectlombok:lombok:1.18.38")
	annotationProcessor("org.projectlombok:lombok:1.18.38")

	implementation("org.springframework.boot:spring-boot-starter-web")
	implementation("org.springframework.boot:spring-boot-starter-actuator")
	implementation("org.springframework.boot:spring-boot-starter-validation")
	
	// DJL (Deep Java Library) for ML inference
	implementation(platform("ai.djl:bom:0.33.0"))
	implementation("ai.djl:api")
	runtimeOnly("ai.djl.pytorch:pytorch-engine")
	implementation("ai.djl.huggingface:tokenizers")
	implementation("ai.djl:model-zoo")

	testImplementation("org.springframework.boot:spring-boot-starter-test")
	testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test> {
	useJUnitPlatform()
	jvmArgs(
		"--sun-misc-unsafe-memory-access=allow",
		"--enable-native-access=ALL-UNNAMED"
	)
}

tasks.withType<JavaExec> {
	jvmArgs(
		"--sun-misc-unsafe-memory-access=allow",
		"--enable-native-access=ALL-UNNAMED"
	)
}
